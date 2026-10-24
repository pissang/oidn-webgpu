//! Experimental native runtime for oidn-web.
//!
//! The TypeScript core (`src/native/program.ts`, compiled by scriptc) plans the
//! network for one input size and records it as a program: buffers with their
//! initial data, WGSL shaders, pipelines, bind groups and the commands of one
//! execution. This crate creates those resources on a wgpu device once and
//! encodes the recorded commands for every execution, so the network shares
//! the caller's device and buffers without any copy through the CPU.

use std::sync::{Mutex, Once};

unsafe extern "C" {
    fn oidn_core_init();
    fn oidn_core_reset();
    fn oidn_core_record_network(
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

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        }
    }
}

/// The scriptc runtime keeps process-global state; one recording at a time.
static CORE: Mutex<()> = Mutex::new(());

fn record(tza: &[u8], config: &[u32]) -> Result<Vec<u8>> {
    static INIT: Once = Once::new();
    let _guard = CORE.lock().map_err(|_| "oidn-web core lock poisoned")?;
    INIT.call_once(|| unsafe { oidn_core_init() });
    let config_bytes: Vec<u8> = config.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut pointer = std::ptr::null();
    let mut size = 0usize;
    let bytes = unsafe {
        oidn_core_record_network(
            tza.as_ptr(),
            tza.len(),
            config_bytes.as_ptr(),
            config_bytes.len(),
            &mut pointer,
            &mut size,
        );
        if pointer.is_null() {
            oidn_core_reset();
            return Err("oidn-web core returned no program".into());
        }
        let bytes = std::slice::from_raw_parts(pointer, size).to_vec();
        oidn_core_reset();
        bytes
    };
    Ok(bytes)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32> {
        let end = self.offset + 4;
        let word = self
            .bytes
            .get(self.offset..end)
            .ok_or("truncated oidn-web program")?;
        self.offset = end;
        Ok(u32::from_le_bytes(word.try_into().unwrap()))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let length = self.u32()? as usize;
        let data = self
            .bytes
            .get(self.offset..self.offset + length)
            .ok_or("truncated oidn-web program")?;
        self.offset += (length + 3) & !3;
        Ok(data)
    }

    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|error| error.to_string())
    }
}

struct BufferDescription {
    size: u64,
    usage: u32,
    data: Vec<u8>,
    label: String,
}

struct BindGroupDescription {
    pipeline: usize,
    group: u32,
    entries: Vec<[u32; 4]>,
}

enum Command {
    Dispatch { pipeline: usize, bind_group: usize, x: u32, y: u32, z: u32 },
    Copy { source: usize, source_offset: u64, destination: usize, destination_offset: u64, size: u64 },
}

enum Operation {
    Submit(Vec<Command>),
    Write { buffer: usize, offset: u64, data: Vec<u8> },
}

struct Program {
    inputs: Vec<usize>,
    output: usize,
    width: u32,
    height: u32,
    output_channels: u32,
    buffers: Vec<BufferDescription>,
    shaders: Vec<String>,
    pipelines: Vec<(usize, String)>,
    bind_groups: Vec<BindGroupDescription>,
    operations: Vec<Operation>,
}

impl Program {
    fn parse(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { bytes, offset: 0 };
        match reader.u32()? {
            PROGRAM_MAGIC => {}
            ERROR_MAGIC => return Err(reader.string()?),
            magic => return Err(format!("unexpected oidn-web program magic {magic:#x}")),
        }
        let version = reader.u32()?;
        if version != 1 {
            return Err(format!("unsupported oidn-web program version {version}"));
        }
        let input_count = reader.u32()?;
        let inputs = (0..input_count)
            .map(|_| reader.u32().map(|id| id as usize))
            .collect::<Result<_>>()?;
        let output = reader.u32()? as usize;
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
            bind_groups.push(BindGroupDescription { pipeline, group, entries });
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
                            kind => return Err(format!("unknown oidn-web command {kind}")),
                        });
                    }
                    operations.push(Operation::Submit(commands));
                }
                1 => operations.push(Operation::Write {
                    buffer: reader.u32()? as usize,
                    offset: reader.u32()? as u64,
                    data: reader.bytes()?.to_vec(),
                }),
                kind => return Err(format!("unknown oidn-web operation {kind}")),
            }
        }
        Ok(Self {
            inputs,
            output,
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
                matrices.push([component, result, properties.m_size, properties.n_size, properties.k_size]);
            }
        }
    }
    // wgpu reports Metal's possible range (4-64); Apple GPUs always run
    // 32-wide simdgroups, which the matrix kernels rely on.
    let (subgroup_min, subgroup_max) =
        if info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple") {
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

/// One planned network for a fixed input size, ready to encode.
pub struct Network {
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    operations: Vec<Operation>,
    inputs: Vec<usize>,
    output: usize,
    width: u32,
    height: u32,
    output_channels: u32,
    /// Time spent in the TypeScript planner, for diagnostics.
    pub planning_ms: f64,
}

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
    ) -> Result<Self> {
        Self::create(device, adapter, tza, config, inputs, false)
    }

    /// Like [`Network::new`], but compiles the shaders without runtime bounds
    /// checks and without zero-initializing workgroup memory. Naga's checks
    /// make the convolutions about four times slower on Metal.
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
    ) -> Result<Self> {
        Self::create(device, adapter, tza, config, inputs, true)
    }

    fn create(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        tza: &[u8],
        config: &NetworkConfig,
        inputs: Option<&[wgpu::Buffer]>,
        unchecked: bool,
    ) -> Result<Self> {
        let mut words = vec![1, config.width, config.height, config.precision as u32, config.kernel as u32];
        words.extend(capability_words(device, adapter));
        words.push(match config.double_buffer {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        });
        words.push(config.f16_accumulation as u32);
        words.push(!config.sub_pixel as u32);
        let started = std::time::Instant::now();
        let program = Program::parse(&record(tza, &words)?)?;
        let planning_ms = started.elapsed().as_secs_f64() * 1e3;
        if let Some(inputs) = inputs {
            if inputs.len() != program.inputs.len() {
                return Err(format!(
                    "the model expects {} input buffers, got {}",
                    program.inputs.len(),
                    inputs.len()
                ));
            }
        }
        let buffers = program
            .buffers
            .iter()
            .enumerate()
            .map(|(id, description)| {
                if let Some(position) = program.inputs.iter().position(|&input| input == id) {
                    if let Some(inputs) = inputs {
                        return inputs[position].clone();
                    }
                }
                let usage = wgpu::BufferUsages::from_bits_truncate(description.usage);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&description.label),
                    size: description.size,
                    usage,
                    mapped_at_creation: !description.data.is_empty(),
                });
                if !description.data.is_empty() {
                    buffer
                        .get_mapped_range_mut(..)
                        .unwrap()
                        .copy_from_slice(&description.data);
                    buffer.unmap();
                }
                buffer
            })
            .collect::<Vec<_>>();
        let modules = program
            .shaders
            .iter()
            .map(|code| {
                let descriptor = wgpu::ShaderModuleDescriptor {
                    label: Some("oidn-web"),
                    source: wgpu::ShaderSource::Wgsl(code.as_str().into()),
                };
                if unchecked {
                    // SAFETY: the caller of `new_unchecked` vouches for the kernels.
                    unsafe {
                        device.create_shader_module_trusted(
                            descriptor,
                            wgpu::ShaderRuntimeChecks::unchecked(),
                        )
                    }
                } else {
                    device.create_shader_module(descriptor)
                }
            })
            .collect::<Vec<_>>();
        let pipelines = program
            .pipelines
            .iter()
            .map(|(shader, entry_point)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("oidn-web"),
                    layout: None,
                    module: &modules[*shader],
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        // Every kernel writes workgroup memory before reading it.
                        zero_initialize_workgroup_memory: !unchecked,
                        ..Default::default()
                    },
                    cache: None,
                })
            })
            .collect::<Vec<_>>();
        let bind_groups = program
            .bind_groups
            .iter()
            .map(|group| {
                let entries = group
                    .entries
                    .iter()
                    .map(|&[binding, buffer, offset, size]| wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffers[buffer as usize],
                            offset: offset as u64,
                            size: std::num::NonZeroU64::new(size as u64),
                        }),
                    })
                    .collect::<Vec<_>>();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("oidn-web"),
                    layout: &pipelines[group.pipeline].get_bind_group_layout(group.group),
                    entries: &entries,
                })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            buffers,
            pipelines,
            bind_groups,
            operations: program.operations,
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

    /// Encodes one execution. Recorded buffer writes are issued on `queue`
    /// immediately, so they take effect before the encoder is submitted.
    pub fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        for operation in &self.operations {
            match operation {
                Operation::Write { buffer, offset, data } => {
                    queue.write_buffer(&self.buffers[*buffer], *offset, data);
                }
                Operation::Submit(commands) => {
                    // Lifetime-free passes let copies end a pass inside the loop.
                    let mut pass: Option<wgpu::ComputePass<'static>> = None;
                    for command in commands {
                        match *command {
                            Command::Dispatch { pipeline, bind_group, x, y, z } => {
                                let pass = pass.get_or_insert_with(|| {
                                    encoder
                                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                            label: Some("oidn-web"),
                                            timestamp_writes: None,
                                        })
                                        .forget_lifetime()
                                });
                                pass.set_pipeline(&self.pipelines[pipeline]);
                                pass.set_bind_group(0, &self.bind_groups[bind_group], &[]);
                                pass.dispatch_workgroups(x, y, z);
                            }
                            Command::Copy { source, source_offset, destination, destination_offset, size } => {
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
