//! GPU denoising on a host-owned wgpu device and image buffers.
//!
//! Load a [`Model`], create a [`Denoiser`], then prepare a fixed image size and
//! buffers. [`PreparedDenoise::encode`] appends work to the host's encoder;
//! the host submits it. Images are tightly packed RGBA f32 storage buffers.
//! The API is experimental. See `examples/host_buffers.rs` for a complete host.

/// Internal C ABI implementation used by interop examples; outside Rust semver guarantees.
#[doc(hidden)]
pub mod capi;
/// Internal GPU C ABI implementation used by interop examples; outside Rust semver guarantees.
#[doc(hidden)]
pub mod gpu_capi;
mod denoiser;
pub use denoiser::*;
/// The exact wgpu crate used by the GPU API. Hosts may use this re-export to
/// avoid incompatible Device and Buffer types from another wgpu dependency.
pub use wgpu;

use std::collections::HashMap;
use std::sync::{Mutex, Once};

unsafe extern "C" {
    fn oidnw_core_init();
    fn oidnw_core_reset();
    fn oidnw_core_inspect_model(tza: *const u8, tza_length: usize, output: *mut *const u8, size: *mut usize);
    fn oidnw_core_record_network(
        tza: *const u8,
        tza_length: usize,
        config: *const u8,
        config_length: usize,
        output: *mut *const u8,
        size: *mut usize,
    );
    fn oidnw_core_record_denoise(
        tza: *const u8,
        tza_length: usize,
        config: *const u8,
        config_length: usize,
        output: *mut *const u8,
        size: *mut usize,
    );
}

const PROGRAM_MAGIC: u32 = 0x4e44_494f;
const ERROR_MAGIC: u32 = 0x4f52_5245;

/// Result for the public denoising API.
pub type Result<T> = std::result::Result<T, Error>;
/// Result for the low-level experimental planner API.
pub type PlanResult<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precision {
    Auto = 0,
    Fp32 = 1,
    Fp16 = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Auto = 0,
    ImplicitGemm = 1,
    Direct = 2,
    SubgroupMatrix = 3,
}

#[derive(Clone, Debug)]
pub struct NetworkConfig {
    pub width: u32,
    pub height: u32,
    pub precision: Precision,
    pub kernel: Kernel,
    /// Double-buffered subgroup-matrix staging; `None` uses the core's default (off).
    pub double_buffer: Option<bool>,
    /// Subgroup-matrix accumulation in f16 instead of f32.
    pub f16_accumulation: bool,
    /// Sub-pixel decomposition of the subgroup-matrix decoders.
    pub sub_pixel: bool,
    /// Records a whole denoising (tiles, input and output processing) of a
    /// `width` x `height` RGBA f32 image instead of the bare network.
    pub denoise: Option<DenoiseConfig>,
}

#[derive(Clone, Debug)]
pub struct DenoiseConfig {
    pub hdr: bool,
    /// HDR transfer: PU (false) or log (true, RTLightmap models).
    pub log_transfer: bool,
    /// Maximum output tile edge; 0 denoises the whole image as one tile.
    pub max_tile_size: u32,
    /// Scale applied to HDR color before the transfer (1 when 0).
    pub input_scale: f32,
}

impl NetworkConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            precision: Precision::Auto,
            kernel: Kernel::Auto,
            double_buffer: None,
            f16_accumulation: false,
            sub_pixel: true,
            denoise: None,
        }
    }
}

/// The scriptc runtime keeps process-global state; one recording at a time.
static CORE: Mutex<()> = Mutex::new(());

fn record(tza: &[u8], config: &[u32], denoise: bool) -> PlanResult<Vec<u8>> {
    let config_bytes: Vec<u8> = config.iter().flat_map(|word| word.to_le_bytes()).collect();
    core_call(|pointer, size| unsafe {
        let entry = if denoise {
            oidnw_core_record_denoise
        } else {
            oidnw_core_record_network
        };
        entry(
            tza.as_ptr(),
            tza.len(),
            config_bytes.as_ptr(),
            config_bytes.len(),
            pointer,
            size,
        );
    })
}

fn core_call(call: impl FnOnce(*mut *const u8, *mut usize)) -> PlanResult<Vec<u8>> {
    static INIT: Once = Once::new();
    let _guard = CORE.lock().map_err(|_| "oidn-webgpu core lock poisoned")?;
    INIT.call_once(|| unsafe { oidnw_core_init() });
    let mut pointer = std::ptr::null();
    let mut size = 0usize;
    call(&mut pointer, &mut size);
    let bytes = unsafe {
        if pointer.is_null() {
            oidnw_core_reset();
            return Err("oidn-webgpu core returned no program".into());
        }
        let bytes = std::slice::from_raw_parts(pointer, size).to_vec();
        oidnw_core_reset();
        bytes
    };
    Ok(bytes)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> PlanResult<u32> {
        let end = self.offset + 4;
        let word = self
            .bytes
            .get(self.offset..end)
            .ok_or("truncated oidn-webgpu program")?;
        self.offset = end;
        Ok(u32::from_le_bytes(word.try_into().unwrap()))
    }

    fn bytes(&mut self) -> PlanResult<&'a [u8]> {
        let length = self.u32()? as usize;
        let data = self
            .bytes
            .get(self.offset..self.offset + length)
            .ok_or("truncated oidn-webgpu program")?;
        self.offset += (length + 3) & !3;
        Ok(data)
    }

    fn string(&mut self) -> PlanResult<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|error| error.to_string())
    }
}

struct BufferDescription {
    size: u64,
    usage: u32,
    data: Vec<u8>,
    label: String,
}

#[derive(Clone)]
struct BindGroupDescription {
    pipeline: usize,
    group: u32,
    entries: Vec<[u32; 4]>,
}

enum Command {
    Dispatch {
        pipeline: usize,
        bind_group: usize,
        x: u32,
        y: u32,
        z: u32,
    },
    Copy {
        source: usize,
        source_offset: u64,
        destination: usize,
        destination_offset: u64,
        size: u64,
    },
}

enum Operation {
    Submit(Vec<Command>),
    Write { buffer: usize, offset: u64, data: Vec<u8> },
}

#[derive(Default)]
struct ProgramMetadata {
    precision: u32,
    tile_count: u32,
    kernels: u32,
    frame: Option<usize>,
    weights: Vec<usize>,
}

struct Program {
    inputs: Vec<usize>,
    output: usize,
    output_external: bool,
    width: u32,
    height: u32,
    output_channels: u32,
    buffers: Vec<BufferDescription>,
    shaders: Vec<String>,
    pipelines: Vec<(usize, String)>,
    bind_groups: Vec<BindGroupDescription>,
    operations: Vec<Operation>,
    metadata: Option<ProgramMetadata>,
}

impl Program {
    fn parse(bytes: &[u8]) -> PlanResult<Self> {
        let mut reader = Reader { bytes, offset: 0 };
        match reader.u32()? {
            PROGRAM_MAGIC => {}
            ERROR_MAGIC => return Err(reader.string()?),
            magic => return Err(format!("unexpected oidn-webgpu program magic {magic:#x}")),
        }
        let version = reader.u32()?;
        if version != 2 {
            return Err(format!("unsupported oidn-webgpu program version {version}"));
        }
        let input_count = reader.u32()?;
        let inputs = (0..input_count)
            .map(|_| reader.u32().map(|id| id as usize))
            .collect::<PlanResult<_>>()?;
        let output = reader.u32()? as usize;
        let output_external = reader.u32()? == 1;
        let width = reader.u32()?;
        let height = reader.u32()?;
        let output_channels = reader.u32()?;
        let mut buffers = Vec::new();
        for _ in 0..reader.u32()? {
            buffers.push(BufferDescription {
                size: reader.u32()? as u64,
                usage: reader.u32()?,
                data: reader.bytes()?.to_vec(),
                label: reader.string()?,
            });
        }
        let mut shaders = Vec::new();
        for _ in 0..reader.u32()? {
            shaders.push(reader.string()?);
        }
        let mut pipelines = Vec::new();
        for _ in 0..reader.u32()? {
            pipelines.push((reader.u32()? as usize, reader.string()?));
        }
        let mut bind_groups = Vec::new();
        for _ in 0..reader.u32()? {
            let pipeline = reader.u32()? as usize;
            let group = reader.u32()?;
            let mut entries = Vec::new();
            for _ in 0..reader.u32()? {
                entries.push([reader.u32()?, reader.u32()?, reader.u32()?, reader.u32()?]);
            }
            bind_groups.push(BindGroupDescription {
                pipeline,
                group,
                entries,
            });
        }
        let mut operations = Vec::new();
        for _ in 0..reader.u32()? {
            match reader.u32()? {
                0 => {
                    let mut commands = Vec::new();
                    for _ in 0..reader.u32()? {
                        commands.push(match reader.u32()? {
                            0 => Command::Dispatch {
                                pipeline: reader.u32()? as usize,
                                bind_group: reader.u32()? as usize,
                                x: reader.u32()?,
                                y: reader.u32()?,
                                z: reader.u32()?,
                            },
                            1 => Command::Copy {
                                source: reader.u32()? as usize,
                                source_offset: reader.u32()? as u64,
                                destination: reader.u32()? as usize,
                                destination_offset: reader.u32()? as u64,
                                size: reader.u32()? as u64,
                            },
                            kind => return Err(format!("unknown oidn-webgpu command {kind}")),
                        });
                    }
                    operations.push(Operation::Submit(commands));
                }
                1 => operations.push(Operation::Write {
                    buffer: reader.u32()? as usize,
                    offset: reader.u32()? as u64,
                    data: reader.bytes()?.to_vec(),
                }),
                kind => return Err(format!("unknown oidn-webgpu operation {kind}")),
            }
        }
        let metadata = if reader.offset < bytes.len() {
            if reader.u32()? != 0x4154_454d || reader.u32()? != 1 {
                return Err("unsupported program metadata".into());
            }
            let precision = reader.u32()?;
            let tile_count = reader.u32()?;
            let kernels = reader.u32()?;
            let frame = match reader.u32()? {
                u32::MAX => None,
                id => Some(id as usize),
            };
            let weights = (0..reader.u32()?)
                .map(|_| reader.u32().map(|id| id as usize))
                .collect::<PlanResult<Vec<_>>>()?;
            if frame.is_some_and(|id| id >= buffers.len()) || weights.iter().any(|&id| id >= buffers.len()) {
                return Err("invalid program metadata buffer id".into());
            }
            Some(ProgramMetadata {
                precision,
                tile_count,
                kernels,
                frame,
                weights,
            })
        } else {
            None
        };
        Ok(Self {
            metadata,
            inputs,
            output,
            output_external,
            width,
            height,
            output_channels,
            buffers,
            shaders,
            pipelines,
            bind_groups,
            operations,
        })
    }
}

/// Device capabilities passed to the planner (config words 5..).
fn capability_words(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Vec<u32> {
    let features = device.features();
    let limits = device.limits();
    let info = adapter.get_info();
    let mut bits = 0;
    if features.contains(wgpu::Features::SHADER_F16) {
        bits |= 1;
    }
    if features.contains(wgpu::Features::SUBGROUP) {
        bits |= 2;
    }
    let mut matrices = Vec::new();
    if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        bits |= 8;
        let scalar = |scalar| match scalar {
            wgpu::CooperativeScalarType::F32 => Some(0),
            wgpu::CooperativeScalarType::F16 => Some(1),
            _ => None,
        };
        for properties in adapter.cooperative_matrix_properties() {
            if let (Some(component), Some(result)) = (scalar(properties.ab_type), scalar(properties.cr_type)) {
                matrices.push([
                    component,
                    result,
                    properties.m_size,
                    properties.n_size,
                    properties.k_size,
                ]);
            }
        }
    }
    // wgpu reports Metal's possible range (4-64); Apple GPUs always run
    // 32-wide simdgroups, which the matrix kernels rely on.
    let (subgroup_min, subgroup_max) = if info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple") {
        (32, 32)
    } else {
        (info.subgroup_min_size, info.subgroup_max_size)
    };
    let mut words = vec![
        bits,
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroups_per_dimension,
        subgroup_min,
        subgroup_max,
        matrices.len() as u32,
    ];
    words.extend(matrices.into_iter().flatten());
    words
}

/// Planner config words (see `src/native/program.ts`).
fn config_words(device: &wgpu::Device, adapter: &wgpu::Adapter, config: &NetworkConfig) -> Vec<u32> {
    let mut words = vec![
        1,
        config.width,
        config.height,
        config.precision as u32,
        config.kernel as u32,
    ];
    words.extend(capability_words(device, adapter));
    words.push(match config.double_buffer {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    });
    words.push(config.f16_accumulation as u32);
    words.push(!config.sub_pixel as u32);
    if let Some(denoise) = &config.denoise {
        words.push(denoise.hdr as u32);
        words.push(denoise.log_transfer as u32);
        words.push(denoise.max_tile_size);
        words.push(denoise.input_scale.to_bits());
    }
    words
}

fn plan(device: &wgpu::Device, adapter: &wgpu::Adapter, tza: &[u8], config: &NetworkConfig) -> PlanResult<Program> {
    Program::parse(&record(
        tza,
        &config_words(device, adapter, config),
        config.denoise.is_some(),
    )?)
}

/// Plans `config` without creating GPU resources and returns the number of
/// input images (color, albedo, normal) of the model.
pub fn plan_inputs(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    tza: &[u8],
    config: &NetworkConfig,
) -> PlanResult<usize> {
    Ok(plan(device, adapter, tza, config)?.inputs.len())
}

/// One planned network for a fixed input size, ready to encode.
pub struct Network {
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_descriptions: Vec<BindGroupDescription>,
    buffer_offsets: Vec<u64>,
    metadata: Option<ProgramMetadata>,
    operations: Vec<Operation>,
    inputs: Vec<usize>,
    output: usize,
    width: u32,
    height: u32,
    output_channels: u32,
    /// Time spent in the TypeScript planner, for diagnostics.
    pub planning_ms: f64,
}

// Cache actual pipeline handles, including their auto-generated layouts, so
// new image sizes and buffer bindings can reuse them on this device.
type WeightMap = HashMap<(u32, Vec<u8>), wgpu::Buffer>;
struct ResourceCaches<'a> {
    pipelines: &'a mut PipelineMap,
    weights: &'a mut WeightMap,
}

type PipelineMap = HashMap<(String, String, bool), wgpu::ComputePipeline>;

impl Network {
    /// Plans the network for `config` and creates its resources on `device`.
    ///
    /// `inputs`, when given, replaces the network's own input buffers; each
    /// must hold `width * height` RGBA f32 pixels in the network's transfer
    /// space and allow `STORAGE` use.
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        tza: &[u8],
        config: &NetworkConfig,
        inputs: Option<&[wgpu::Buffer]>,
    ) -> PlanResult<Self> {
        Self::create(device, adapter, tza, config, inputs, None, false)
    }

    /// Like [`Network::new`], but compiles the shaders without runtime bounds
    /// checks and without zero-initializing workgroup memory. The performance
    /// difference depends on the GPU and generated convolution kernels.
    ///
    /// # Safety
    ///
    /// The generated kernels must not access memory out of bounds for this
    /// configuration. They are written to stay in bounds and validated in
    /// browsers, whose robustness transforms would hide such accesses, so a
    /// kernel bug here can read or write arbitrary GPU memory.
    pub unsafe fn new_unchecked(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        tza: &[u8],
        config: &NetworkConfig,
        inputs: Option<&[wgpu::Buffer]>,
    ) -> PlanResult<Self> {
        Self::create(device, adapter, tza, config, inputs, None, true)
    }

    /// Plans with `config` (which must request a denoising) and binds the
    /// caller's image buffers: `inputs` (color, then albedo and normal for
    /// auxiliary models) and `output`, each `width * height` RGBA f32 pixels
    /// with `STORAGE` use. `unchecked` has the contract of
    /// [`Network::new_unchecked`].
    ///
    /// # Safety
    ///
    /// When `unchecked` is true, as for [`Network::new_unchecked`].
    pub unsafe fn with_buffers(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        tza: &[u8],
        config: &NetworkConfig,
        inputs: &[wgpu::Buffer],
        output: &wgpu::Buffer,
        unchecked: bool,
    ) -> PlanResult<Self> {
        Self::create(device, adapter, tza, config, Some(inputs), Some(output), unchecked)
    }

    fn create(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        tza: &[u8],
        config: &NetworkConfig,
        inputs: Option<&[wgpu::Buffer]>,
        output: Option<&wgpu::Buffer>,
        unchecked: bool,
    ) -> PlanResult<Self> {
        let started = std::time::Instant::now();
        let program = plan(device, adapter, tza, config)?;
        let planning_ms = started.elapsed().as_secs_f64() * 1e3;
        Self::from_program(
            device,
            program,
            inputs,
            output,
            unchecked,
            ResourceCaches {
                pipelines: &mut PipelineMap::new(),
                weights: &mut WeightMap::new(),
            },
            planning_ms,
        )
    }

    fn from_program(
        device: &wgpu::Device,
        program: Program,
        inputs: Option<&[wgpu::Buffer]>,
        output: Option<&wgpu::Buffer>,
        unchecked: bool,
        caches: ResourceCaches<'_>,
        planning_ms: f64,
    ) -> PlanResult<Self> {
        let ResourceCaches {
            pipelines: cache,
            weights,
        } = caches;
        if output.is_some() && !program.output_external {
            return Err("the network program has no external output".into());
        }
        if let Some(inputs) = inputs
            && inputs.len() != program.inputs.len()
        {
            return Err(format!(
                "the model expects {} input buffers, got {}",
                program.inputs.len(),
                inputs.len()
            ));
        }
        let mut buffers = program
            .buffers
            .iter()
            .enumerate()
            .map(|(id, description)| {
                if let Some(position) = program.inputs.iter().position(|&input| input == id)
                    && let Some(inputs) = inputs
                {
                    return inputs[position].clone();
                }
                if id == program.output
                    && let Some(output) = output
                {
                    return output.clone();
                }
                let immutable = program.metadata.as_ref().is_some_and(|m| m.weights.contains(&id));
                let key = immutable.then(|| (description.usage, description.data.clone()));
                if let Some(key) = &key
                    && let Some(buffer) = weights.get(key)
                {
                    return buffer.clone();
                }
                let mut usage = wgpu::BufferUsages::from_bits_truncate(description.usage);
                if program
                    .operations
                    .iter()
                    .any(|op| matches!(op, Operation::Write { buffer, .. } if *buffer == id))
                {
                    usage |= wgpu::BufferUsages::COPY_DST;
                }
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&description.label),
                    size: description.size,
                    usage,
                    mapped_at_creation: !description.data.is_empty(),
                });
                if !description.data.is_empty() {
                    buffer
                        .get_mapped_range_mut(..description.data.len() as u64)
                        .unwrap()
                        .copy_from_slice(&description.data);
                    buffer.unmap();
                }
                if let Some(key) = key {
                    weights.insert(key, buffer.clone());
                }
                buffer
            })
            .collect::<Vec<_>>();
        let pipelines = program
            .pipelines
            .iter()
            .map(|(shader, entry_point)| {
                let code = &program.shaders[*shader];
                let key = (code.clone(), entry_point.clone(), unchecked);
                cache
                    .entry(key)
                    .or_insert_with(|| {
                        let descriptor = wgpu::ShaderModuleDescriptor {
                            label: Some("oidn-webgpu"),
                            source: wgpu::ShaderSource::Wgsl(code.as_str().into()),
                        };
                        let module = if unchecked {
                            // SAFETY: the caller of `new_unchecked` vouches for the kernels.
                            unsafe {
                                device.create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked())
                            }
                        } else {
                            device.create_shader_module(descriptor)
                        };
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("oidn-webgpu"),
                            layout: None,
                            module: &module,
                            entry_point: Some(entry_point),
                            compilation_options: wgpu::PipelineCompilationOptions {
                                // Every kernel writes workgroup memory before reading it.
                                zero_initialize_workgroup_memory: !unchecked,
                                ..Default::default()
                            },
                            cache: None,
                        })
                    })
                    .clone()
            })
            .collect::<Vec<_>>();
        let buffer_offsets = vec![0; buffers.len()];
        let mut descriptions = program.bind_groups;
        for group in &mut descriptions {
            for entry in &mut group.entries {
                if entry[3] == 0 {
                    entry[3] = (program.buffers[entry[1] as usize].size - entry[2] as u64) as u32;
                }
            }
        }
        let bind_groups = create_bind_groups(device, &pipelines, &descriptions, &buffers, &buffer_offsets);
        // A queue write would move ahead of all encoded commands at submit.
        // Immutable staging copies preserve the program's interleaved writes
        // and dispatches, including repeated encodes in one command buffer.
        let mut operations = Vec::new();
        for operation in program.operations {
            match operation {
                Operation::Write { buffer, offset, data } => {
                    if data.is_empty() {
                        continue;
                    }
                    if offset % 4 != 0 || data.len() % 4 != 0 {
                        return Err("unaligned recorded buffer write".into());
                    }
                    let staging = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("oidn-webgpu recorded write"),
                        size: data.len() as u64,
                        usage: wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    });
                    staging.get_mapped_range_mut(..).unwrap().copy_from_slice(&data);
                    staging.unmap();
                    let source = buffers.len();
                    buffers.push(staging);
                    operations.push(Operation::Submit(vec![Command::Copy {
                        source,
                        source_offset: 0,
                        destination: buffer,
                        destination_offset: offset,
                        size: data.len() as u64,
                    }]));
                }
                op => operations.push(op),
            }
        }
        Ok(Self {
            buffers,
            pipelines,
            bind_groups,
            bind_group_descriptions: descriptions,
            buffer_offsets,
            metadata: program.metadata,
            operations,
            inputs: program.inputs,
            output: program.output,
            width: program.width,
            height: program.height,
            output_channels: program.output_channels,
            planning_ms,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn output_channels(&self) -> u32 {
        self.output_channels
    }

    /// The input buffers the network reads (RGBA f32 per pixel).
    pub fn input_buffers(&self) -> impl Iterator<Item = &wgpu::Buffer> {
        self.inputs.iter().map(|&id| &self.buffers[id])
    }

    /// The output buffer (RGBA f32 per pixel) written by every execution.
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[self.output]
    }

    /// Encodes one execution without submitting or writing to the queue.
    /// The queue argument is retained for compatibility with the low-level API.
    pub fn encode(&self, _queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.encode_commands(encoder);
    }

    fn encode_commands(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_profiled(encoder, None);
    }

    /// Writes optional timestamps at the first and last compute pass. The
    /// device must enable TIMESTAMP_QUERY and indices must name unused slots.
    pub fn encode_with_timestamps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        begin: Option<u32>,
        end: Option<u32>,
    ) {
        self.encode_profiled(encoder, Some((queries, begin, end)));
    }

    fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profile: Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)>,
    ) {
        let pass_count: usize = if profile.is_some() {
            self.operations
                .iter()
                .map(|operation| {
                    let mut count = 0;
                    let mut in_pass = false;
                    if let Operation::Submit(commands) = operation {
                        for command in commands {
                            if matches!(command, Command::Dispatch { .. }) {
                                if !in_pass {
                                    count += 1;
                                }
                                in_pass = true;
                            } else {
                                in_pass = false;
                            }
                        }
                    }
                    count
                })
                .sum()
        } else {
            0
        };
        let mut pass_index = 0;
        for operation in &self.operations {
            match operation {
                Operation::Write { .. } => unreachable!("writes are lowered during preparation"),
                Operation::Submit(commands) => {
                    // Lifetime-free passes let copies end a pass inside the loop.
                    let mut pass: Option<wgpu::ComputePass<'static>> = None;
                    for command in commands {
                        match *command {
                            Command::Dispatch {
                                pipeline,
                                bind_group,
                                x,
                                y,
                                z,
                            } => {
                                let pass = pass.get_or_insert_with(|| {
                                    let timestamp_writes =
                                        profile.and_then(|(queries, begin, end)| {
                                            let beginning_of_pass_write_index =
                                                if pass_index == 0 { begin } else { None };
                                            let end_of_pass_write_index =
                                                if pass_index + 1 == pass_count {
                                                    end
                                                } else {
                                                    None
                                                };
                                            (beginning_of_pass_write_index.is_some()
                                                || end_of_pass_write_index.is_some())
                                            .then_some(wgpu::ComputePassTimestampWrites {
                                                query_set: queries,
                                                beginning_of_pass_write_index,
                                                end_of_pass_write_index,
                                            })
                                        });
                                    pass_index += 1;
                                    encoder
                                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                            label: Some("oidn-webgpu"),
                                            timestamp_writes,
                                        })
                                        .forget_lifetime()
                                });
                                pass.set_pipeline(&self.pipelines[pipeline]);
                                pass.set_bind_group(0, &self.bind_groups[bind_group], &[]);
                                pass.dispatch_workgroups(x, y, z);
                            }
                            Command::Copy {
                                source,
                                source_offset,
                                destination,
                                destination_offset,
                                size,
                            } => {
                                drop(pass.take());
                                encoder.copy_buffer_to_buffer(
                                    &self.buffers[source],
                                    source_offset,
                                    &self.buffers[destination],
                                    destination_offset,
                                    size,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// Base offsets are nonzero only for external image views. Binding lengths
// always come from the recorded plan, never the rest of a host allocation.
fn create_bind_groups(
    device: &wgpu::Device,
    pipelines: &[wgpu::ComputePipeline],
    descriptions: &[BindGroupDescription],
    buffers: &[wgpu::Buffer],
    offsets: &[u64],
) -> Vec<wgpu::BindGroup> {
    descriptions
        .iter()
        .map(|group| {
            let entries: Vec<_> = group
                .entries
                .iter()
                .map(|&[binding, buffer, offset, size]| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers[buffer as usize],
                        offset: offsets[buffer as usize] + offset as u64,
                        size: std::num::NonZeroU64::new(size as u64),
                    }),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oidn-webgpu"),
                layout: &pipelines[group.pipeline].get_bind_group_layout(group.group),
                entries: &entries,
            })
        })
        .collect()
}
