use crate::{DenoiseConfig, Kernel, Network, NetworkConfig, PipelineMap, Reader, Result, WeightMap};
use std::{
    fmt,
    sync::{Arc, Mutex, Weak},
    time::Instant,
};

/// Device limit that prevented preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    ProgramBufferSize,
    BufferSize,
    StorageBindingSize,
}

/// Loading, preparation and CPU argument errors. GPU execution errors are
/// delivered through the host's wgpu error handling after commands are submitted.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    InvalidArgument(String),
    Model(String),
    UnsupportedDevice(String),
    Gpu(String),
    BufferTooSmall {
        image: &'static str,
        required: u64,
        available: u64,
    },
    LimitExceeded {
        resource: String,
        limit: Limit,
        required: u64,
        available: u64,
    },
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::Model(m) => write!(f, "model: {m}"),
            Self::UnsupportedDevice(m) => write!(f, "unsupported device: {m}"),
            Self::Gpu(m) => write!(f, "GPU: {m}"),
            Self::BufferTooSmall {
                image,
                required,
                available,
            } => write!(f, "{image} needs {required} bytes, has {available}"),
            Self::LimitExceeded {
                resource,
                limit,
                required,
                available,
            } => write!(
                f,
                "{resource} requires {required} bytes, {limit:?} allows {available}; request higher limits or use smaller tiles/images"
            ),
        }
    }
}
impl std::error::Error for Error {}

// Models keep weak references so merely retaining CPU weights does not keep a
// GPU device alive. Denoisers and prepared instances own active cache contexts.
struct SharedWeights {
    device: wgpu::Device,
    buffers: Mutex<WeightMap>,
}
/// Validated, owned CPU weights. Clones share immutable GPU weights per
/// device while denoisers or prepared instances using them remain alive.
#[derive(Clone)]
pub struct Model {
    weights: Arc<[u8]>,
    inputs: usize,
    devices: Arc<Mutex<Vec<Weak<SharedWeights>>>>,
}
impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("input_count", &self.inputs)
            .field("weight_bytes", &self.weights.len())
            .finish()
    }
}
impl Model {
    /// Validates and owns an OIDN TZA file without creating a GPU device.
    /// Clones share CPU weights and device-specific immutable GPU weights.
    pub fn from_tza(weights: &[u8]) -> Result<Self> {
        let bytes = crate::core_call(|pointer, size| unsafe {
            crate::oidnw_core_inspect_model(weights.as_ptr(), weights.len(), pointer, size);
        })
        .map_err(Error::Model)?;
        let mut reader = Reader {
            bytes: &bytes,
            offset: 0,
        };
        let inputs = (|| -> crate::PlanResult<usize> {
            match reader.u32()? {
                crate::ERROR_MAGIC => return Err(reader.string()?),
                0x4c44_4f4d => {}
                _ => return Err("invalid model metadata".into()),
            }
            if reader.u32()? != 1 {
                return Err("unsupported model metadata version".into());
            }
            match reader.u32()? {
                count @ 1..=3 => Ok(count as usize),
                _ => Err("model must read 1 to 3 images".into()),
            }
        })()
        .map_err(Error::Model)?;
        Ok(Self {
            weights: Arc::from(weights),
            inputs,
            devices: Arc::new(Mutex::new(Vec::new())),
        })
    }
    /// 1 = color, 2 = color/albedo, 3 = color/albedo/normal.
    pub fn input_count(&self) -> usize {
        self.inputs
    }
    fn shared_weights(&self, device: &wgpu::Device) -> Result<Arc<SharedWeights>> {
        let mut devices = self
            .devices
            .lock()
            .map_err(|_| Error::Gpu("weight cache lock poisoned".into()))?;
        devices.retain(|entry| entry.strong_count() > 0);
        if let Some(cache) = devices
            .iter()
            .filter_map(Weak::upgrade)
            .find(|cache| &cache.device == device)
        {
            return Ok(cache);
        }
        let cache = Arc::new(SharedWeights {
            device: device.clone(),
            buffers: Mutex::new(WeightMap::new()),
        });
        devices.push(Arc::downgrade(&cache));
        Ok(cache)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transfer {
    #[default]
    Pu,
    Log,
}
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DenoiserOptions {
    /// Linear HDR color, or linear LDR RGB in [0, 1] when false.
    pub hdr: bool,
    /// PU for RT HDR, Log for RTLightmap HDR models.
    pub transfer: Transfer,
    /// Output tile edge rounded down to a multiple of 16; 0 = whole image.
    pub max_tile_size: u32,
}
impl Default for DenoiserOptions {
    fn default() -> Self {
        Self {
            hdr: true,
            transfer: Transfer::Pu,
            max_tile_size: 0,
        }
    }
}
/// Per-execution values. Changing these does not prepare resources again.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    /// Positive finite scale applied before HDR transfer, undone on output.
    /// Ignored for LDR. Exposure is not estimated automatically.
    pub input_scale: f32,
}
impl Default for DenoiseParams {
    fn default() -> Self {
        Self { input_scale: 1.0 }
    }
}

/// A tightly packed RGBA f32 image starting at a byte offset. The binding
/// length is width * height * 16, independent of the host allocation's size.
#[derive(Clone, Copy, Debug)]
pub struct ImageBuffer<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
}
impl<'a> From<&'a wgpu::Buffer> for ImageBuffer<'a> {
    fn from(buffer: &'a wgpu::Buffer) -> Self {
        Self { buffer, offset: 0 }
    }
}
/// Host-owned, unmapped STORAGE buffers from the denoiser's device.
/// Input ranges must not overlap; output must use a separate allocation.
/// Albedo is linear [0, 1]; normals
/// are signed [-1, 1] world/view-space shading normals. Auxiliary alpha is ignored.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseBuffers<'a> {
    pub color: ImageBuffer<'a>,
    pub albedo: Option<ImageBuffer<'a>>,
    pub normal: Option<ImageBuffer<'a>>,
    /// The output preserves color alpha.
    pub output: ImageBuffer<'a>,
}
impl DenoiseBuffers<'_> {
    fn validate(&self, count: usize, bytes: u64, alignment: u64) -> Result<Vec<ImageBuffer<'_>>> {
        if self.albedo.is_some() != (count >= 2) || self.normal.is_some() != (count >= 3) {
            return Err(Error::InvalidArgument(format!(
                "model requires exactly {count} images: color, then albedo, then normal"
            )));
        }
        let all: Vec<_> = [
            ("color", Some(self.color)),
            ("albedo", self.albedo),
            ("normal", self.normal),
            ("output", Some(self.output)),
        ]
        .into_iter()
        .filter_map(|(name, view)| view.map(|v| (name, v)))
        .collect();
        for (i, &(name, view)) in all.iter().enumerate() {
            if !view.buffer.usage().contains(wgpu::BufferUsages::STORAGE) || !view.offset.is_multiple_of(alignment) {
                return Err(Error::InvalidArgument(format!(
                    "{name} needs STORAGE usage and an offset aligned to {alignment} bytes"
                )));
            }
            let available = view.buffer.size().saturating_sub(view.offset);
            if view.offset > view.buffer.size() || bytes > available {
                return Err(Error::BufferTooSmall {
                    image: name,
                    required: bytes,
                    available,
                });
            }
            for &(other_name, other) in &all[..i] {
                if view.buffer == other.buffer
                    && view.offset < other.offset + bytes
                    && other.offset < view.offset + bytes
                {
                    return Err(Error::InvalidArgument(format!("{name} overlaps {other_name}")));
                }
            }
        }
        if all[..all.len() - 1]
            .iter()
            .any(|(_, view)| view.buffer == self.output.buffer)
        {
            return Err(Error::InvalidArgument(
                "output must use a different buffer from inputs: wgpu storage usage applies to the whole allocation"
                    .into(),
            ));
        }
        Ok(all.into_iter().map(|(_, view)| view).collect())
    }
}

/// Convolution paths actually selected by the planner; a network may use more
/// than one (for example matrix convolutions and a direct final RGB layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPath {
    Direct,
    ImplicitGemm,
    SubgroupMatrix,
    Spatial,
    Subgroup,
}
/// Resource sizes and selected execution paths for this prepared instance.
#[derive(Clone, Debug)]
pub struct DenoiseInfo {
    pub size: [u32; 2],
    /// Activation/weight storage; image buffers and output remain f32.
    pub precision: crate::Precision,
    pub kernels: Vec<KernelPath>,
    pub tile_count: u32,
    /// Per-instance GPU buffers, excluding host images and shared weights.
    /// Small transient parameter-upload buffers and driver overhead are excluded.
    pub workspace_bytes: u64,
    /// Immutable weights used by this instance; shared by the same Model/device.
    pub shared_weight_bytes: u64,
    /// CPU planning and weight-packing time for this preparation.
    pub planning_ms: f64,
    /// Total preparation time, including resource creation and shader compilation.
    pub prepare_ms: f64,
    /// Whether shader bounds checks and workgroup initialization are enabled.
    pub runtime_checks: bool,
}

/// Device-bound model and pipeline cache. Preparation reuses pipelines and
/// immutable weights; every prepared instance has a separate workspace.
pub struct Denoiser {
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    model: Model,
    options: DenoiserOptions,
    pipelines: PipelineMap,
    weights: Arc<SharedWeights>,
    unchecked: bool,
}
impl Denoiser {
    pub fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    /// Optional stable acceleration; hosts may additionally opt into wgpu's
    /// experimental cooperative-matrix feature contract before device creation.
    pub fn recommended_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & (wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP)
    }
    /// Baseline; larger images may require higher buffer limits.
    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        model: &Model,
        options: DenoiserOptions,
    ) -> Result<Self> {
        if device.adapter_info() != adapter.get_info() {
            return Err(Error::InvalidArgument("adapter must match device".into()));
        }
        if options.max_tile_size != 0 && !(16..=16384).contains(&options.max_tile_size) {
            return Err(Error::InvalidArgument(
                "max_tile_size must be 0 or in 16..=16384".into(),
            ));
        }
        Ok(Self {
            device: device.clone(),
            adapter: adapter.clone(),
            model: model.clone(),
            options,
            pipelines: PipelineMap::new(),
            weights: model.shared_weights(device)?,
            unchecked: false,
        })
    }
    /// Disables shader bounds checks and workgroup initialization.
    ///
    /// # Safety
    /// Generated kernels must stay in bounds and initialize workgroup data
    /// before reading it for every configuration prepared with this denoiser.
    pub unsafe fn new_unchecked(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        model: &Model,
        options: DenoiserOptions,
    ) -> Result<Self> {
        let mut denoiser = Self::new(device, adapter, model, options)?;
        denoiser.unchecked = true;
        Ok(denoiser)
    }
    /// Plans and allocates a fixed size before interactive rendering. May block
    /// for shader compilation. Instances may outlive this Denoiser; they retain
    /// their image handles, workspace and shared weights.
    pub fn prepare(&mut self, size: [u32; 2], buffers: DenoiseBuffers<'_>) -> Result<PreparedDenoise> {
        let device = &self.device;
        let [width, height] = size;
        if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
            return Err(Error::InvalidArgument("width and height must be in 1..=16384".into()));
        }
        let bytes = width as u64 * height as u64 * 16;
        check_buffer_limits(device, "external image", bytes, true)?;
        let views = buffers.validate(
            self.model.inputs,
            bytes,
            device.limits().min_storage_buffer_offset_alignment as u64,
        )?;
        let input_buffers: Vec<_> = views[..self.model.inputs].iter().map(|v| v.buffer.clone()).collect();
        let config = NetworkConfig {
            kernel: if device
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            {
                Kernel::SubgroupMatrix
            } else {
                Kernel::ImplicitGemm
            },
            denoise: Some(DenoiseConfig {
                hdr: self.options.hdr,
                log_transfer: self.options.transfer == Transfer::Log,
                max_tile_size: self.options.max_tile_size,
                input_scale: 1.0,
            }),
            ..NetworkConfig::new(width, height)
        };
        let started = Instant::now();
        let program = crate::plan(device, &self.adapter, &self.model.weights, &config).map_err(Error::Model)?;
        for buffer in &program.buffers {
            check_buffer_limits(
                device,
                &buffer.label,
                buffer.size,
                buffer.usage & wgpu::BufferUsages::STORAGE.bits() != 0,
            )?;
        }
        let meta = program.metadata.as_ref().ok_or_else(|| {
            Error::Model("core archive lacks execution metadata; rebuild it for this source revision".into())
        })?;
        let info = DenoiseInfo {
            size,
            precision: if meta.precision == 2 {
                crate::Precision::Fp16
            } else {
                crate::Precision::Fp32
            },
            kernels: [
                KernelPath::Direct,
                KernelPath::ImplicitGemm,
                KernelPath::SubgroupMatrix,
                KernelPath::Spatial,
                KernelPath::Subgroup,
            ]
            .into_iter()
            .enumerate()
            .filter_map(|(i, k)| (meta.kernels & (1 << i) != 0).then_some(k))
            .collect(),
            tile_count: meta.tile_count,
            workspace_bytes: 0,
            shared_weight_bytes: 0,
            planning_ms: started.elapsed().as_secs_f64() * 1e3,
            prepare_ms: 0.0,
            runtime_checks: !self.unchecked,
        };
        let mut weights = self
            .weights
            .buffers
            .lock()
            .map_err(|_| Error::Gpu("weight cache lock poisoned".into()))?;
        let network = gpu_checked(device, || {
            Network::from_program(
                device,
                program,
                Some(&input_buffers),
                Some(buffers.output.buffer),
                self.unchecked,
                crate::ResourceCaches {
                    pipelines: &mut self.pipelines,
                    weights: &mut weights,
                },
                info.planning_ms,
            )
            .map_err(Error::Model)
        });
        let network = match network {
            Ok(n) => n,
            Err(e) => {
                self.pipelines.clear();
                weights.clear();
                return Err(e);
            }
        };
        drop(weights);
        let mut prepared = PreparedDenoise {
            device: device.clone(),
            staging: wgpu::util::StagingBelt::new(device.clone(), 256),
            network,
            inputs: self.model.inputs,
            image_bytes: bytes,
            info,
            _weights: self.weights.clone(),
        };
        // Zero-offset host bindings were already created by from_program.
        if views.iter().any(|view| view.offset != 0) {
            prepared.rebind(buffers)?;
        }
        let meta = prepared.network.metadata.as_ref().unwrap();
        let mut seen_weights = Vec::new();
        for &id in &meta.weights {
            let buffer = &prepared.network.buffers[id];
            if !seen_weights.contains(&buffer) {
                prepared.info.shared_weight_bytes += buffer.size();
                seen_weights.push(buffer);
            }
        }
        prepared.info.workspace_bytes = prepared
            .network
            .buffers
            .iter()
            .enumerate()
            .filter(|(id, _)| {
                !prepared.network.inputs.contains(id) && *id != prepared.network.output && !meta.weights.contains(id)
            })
            .map(|(_, b)| b.size())
            .sum();
        prepared.info.prepare_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(prepared)
    }
}
fn check_buffer_limits(device: &wgpu::Device, resource: &str, required: u64, storage: bool) -> Result<()> {
    let limits = device.limits();
    for (limit, available) in [
        (Limit::ProgramBufferSize, u32::MAX as u64),
        (Limit::BufferSize, limits.max_buffer_size),
        (
            Limit::StorageBindingSize,
            if storage {
                limits.max_storage_buffer_binding_size
            } else {
                u64::MAX
            },
        ),
    ] {
        if required > available {
            return Err(Error::LimitExceeded {
                resource: resource.into(),
                limit,
                required,
                available,
            });
        }
    }
    Ok(())
}
fn gpu_checked<T>(device: &wgpu::Device, body: impl FnOnce() -> Result<T>) -> Result<T> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = body();
    let errors = [
        pollster::block_on(validation.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(oom.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(Error::Gpu(error.to_string()));
    }
    result
}

/// A fixed-size workspace. Submit executions in encode order on its device's
/// queue, and end any active pass before encode. Separate instances have
/// independent workspaces. Do not explicitly destroy buffers while in use.
pub struct PreparedDenoise {
    device: wgpu::Device,
    network: Network,
    inputs: usize,
    image_bytes: u64,
    info: DenoiseInfo,
    staging: wgpu::util::StagingBelt,
    _weights: Arc<SharedWeights>,
}
impl PreparedDenoise {
    pub fn size(&self) -> [u32; 2] {
        self.info.size
    }
    pub fn info(&self) -> &DenoiseInfo {
        &self.info
    }
    /// Appends commands only, with a per-execution snapshot of parameters.
    /// Reports invalid CPU parameters before modifying the encoder. GPU/device
    /// errors use the host's wgpu error handling, not this Result.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, params: DenoiseParams) -> Result<()> {
        if !(params.input_scale.is_finite() && params.input_scale > 0.0) {
            return Err(Error::InvalidArgument("input_scale must be positive and finite".into()));
        }
        let frame = self
            .network
            .metadata
            .as_ref()
            .and_then(|meta| meta.frame)
            .ok_or_else(|| Error::Model("missing frame parameter buffer".into()))?;
        self.staging
            .write_buffer(
                encoder,
                &self.network.buffers[frame],
                0,
                std::num::NonZeroU64::new(4).unwrap(),
            )
            .copy_from_slice(&params.input_scale.to_le_bytes());
        self.network.encode_commands(encoder);
        self.staging.finish_and_recall_on_submit(encoder);
        Ok(())
    }
    /// Replaces image ranges at the same size, preserving weights, pipelines
    /// and workspace. Already encoded commands retain previous bindings.
    pub fn rebind(&mut self, buffers: DenoiseBuffers<'_>) -> Result<()> {
        let views = buffers.validate(
            self.inputs,
            self.image_bytes,
            self.device.limits().min_storage_buffer_offset_alignment as u64,
        )?;
        let mut candidate = self.network.buffers.clone();
        let mut offsets = self.network.buffer_offsets.clone();
        for (&id, view) in self
            .network
            .inputs
            .iter()
            .chain(std::iter::once(&self.network.output))
            .zip(views)
        {
            candidate[id] = view.buffer.clone();
            offsets[id] = view.offset;
        }
        let bind_groups = gpu_checked(&self.device, || {
            Ok(crate::create_bind_groups(
                &self.device,
                &self.network.pipelines,
                &self.network.bind_group_descriptions,
                &candidate,
                &offsets,
            ))
        })?;
        self.network.buffers = candidate;
        self.network.buffer_offsets = offsets;
        self.network.bind_groups = bind_groups;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Program;
    use std::time::Duration;

    fn weights() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../weights/rt_hdr_alb_nrm_small.tza"
        ))
        .unwrap()
    }

    #[test]
    fn model_metadata_without_gpu() {
        assert!(matches!(Model::from_tza(b"invalid weights"), Err(Error::Model(_))));
        assert!(matches!(Model::from_tza(&[]), Err(Error::Model(_))));
        assert_eq!(Model::from_tza(&weights()).unwrap().input_count(), 3);
        let color = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../weights/rt_hdr.tza")).unwrap();
        assert_eq!(Model::from_tza(&color).unwrap().input_count(), 1);
    }

    fn download(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, bytes);
        let submission = queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| send.send(result).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
        receive.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        let data = readback.get_mapped_range(..).unwrap().to_vec();
        readback.unmap();
        data
    }

    // GPU tests are explicit so a machine without an adapter does not silently
    // pass them. Run: cargo test --manifest-path native/Cargo.toml -- --ignored
    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn packed_ldr_color_matches_separate_pack_at_odd_image_edges() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../weights/rt_ldr.tza"
        ))
        .unwrap();
        let model = Model::from_tza(&bytes).unwrap();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        for features in [wgpu::Features::empty(), wgpu::Features::SHADER_F16] {
            if !adapter.features().contains(features) {
                continue;
            }
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_features: features,
                    ..Default::default()
                }))
                .unwrap();
            let image = |width: u32, height: u32| -> Vec<u8> {
                (0..height)
                    .flat_map(|y| {
                        (0..width).flat_map(move |x| {
                            let pixel = y.min(16) * 19 + x.min(18);
                            [
                                0.1 + (pixel % 13) as f32 / 32.0,
                                0.2,
                                0.3,
                                pixel as f32 / 1024.0,
                            ]
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                        })
                    })
                    .collect()
            };
            let buffer = |data: &[u8]| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: data.len() as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buffer, 0, data);
                buffer
            };
            let input_bytes = image(19, 17);
            let color = buffer(&input_bytes);
            let output = buffer(&vec![0u8; input_bytes.len()]);
            let padded = buffer(&image(32, 32));
            let reference = Network::new(
                &device,
                &adapter,
                &bytes,
                &NetworkConfig {
                    precision: if features.contains(wgpu::Features::SHADER_F16) {
                        crate::Precision::Fp16
                    } else {
                        crate::Precision::Fp32
                    },
                    ..NetworkConfig::new(32, 32)
                },
                Some(&[padded]),
            )
            .unwrap();
            let mut denoiser = Denoiser::new(
                &device,
                &adapter,
                &model,
                DenoiserOptions {
                    hdr: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut prepared = denoiser
                .prepare(
                    [19, 17],
                    DenoiseBuffers {
                        color: (&color).into(),
                        albedo: None,
                        normal: None,
                        output: (&output).into(),
                    },
                )
                .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            reference.encode(&queue, &mut encoder);
            prepared
                .encode(&mut encoder, DenoiseParams::default())
                .unwrap();
            queue.submit([encoder.finish()]);
            let actual = download(&device, &queue, &output, input_bytes.len() as u64);
            let expected = download(&device, &queue, reference.output_buffer(), 32 * 32 * 16);
            for y in 0..17 {
                for x in 0..19 {
                    for channel in 0..4 {
                        let offset = ((y * 19 + x) * 4 + channel) * 4;
                        let expected = if channel == 3 {
                            &input_bytes[offset..offset + 4]
                        } else {
                            let offset = ((y * 32 + x) * 4 + channel) * 4;
                            &expected[offset..offset + 4]
                        };
                        assert_eq!(
                            &actual[offset..offset + 4],
                            expected,
                            "{features:?} pixel {x},{y}, channel {channel}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn host_buffers_lifecycle_and_recorded_write_order() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: Denoiser::required_features(),
            required_limits: Denoiser::required_limits(),
            ..Default::default()
        }))
        .unwrap();
        let model = Model::from_tza(&weights()).unwrap();
        let mut denoiser = Denoiser::new(&device, &adapter, &model, DenoiserOptions::default()).unwrap();
        let size = [17, 19];
        let bytes = size[0] as u64 * size[1] as u64 * 16;
        let image = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let color = image();
        let albedo = image();
        let normal = image();
        let output = image();
        let output2 = image();
        let bindings = DenoiseBuffers {
            color: (&color).into(),
            albedo: Some((&albedo).into()),
            normal: Some((&normal).into()),
            output: (&output).into(),
        };
        let data: Vec<u8> = (0..size[0] * size[1])
            .flat_map(|p| [0.5 + (p % 7) as f32 * 0.01, 0.3, 0.1, p as f32 / 1024.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        queue.write_buffer(&color, 0, &data);
        let normals: Vec<u8> = (0..size[0] * size[1])
            .flat_map(|_| [0.0f32, 0.0, 1.0, 0.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        queue.write_buffer(&normal, 0, &normals);

        assert!(matches!(
            denoiser.prepare([0, 19], bindings),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            denoiser.prepare(
                size,
                DenoiseBuffers {
                    albedo: None,
                    ..bindings
                }
            ),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            denoiser.prepare(
                size,
                DenoiseBuffers {
                    output: (&color).into(),
                    ..bindings
                }
            ),
            Err(Error::InvalidArgument(_))
        ));
        let short = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(matches!(
            denoiser.prepare(
                size,
                DenoiseBuffers {
                    output: (&short).into(),
                    ..bindings
                }
            ),
            Err(Error::BufferTooSmall { .. })
        ));

        let mut prepared = denoiser.prepare(size, bindings).unwrap();
        let pipelines = denoiser.pipelines.clone();
        let _other_size = denoiser.prepare([16, 16], bindings).unwrap();
        for (key, pipeline) in &pipelines {
            assert_eq!(denoiser.pipelines.get(key), Some(pipeline));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
        queue.submit([encoder.finish()]);
        let expected = download(&device, &queue, &output, bytes);
        assert!(
            expected
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        );
        for (actual, input) in expected.chunks_exact(16).zip(data.chunks_exact(16)) {
            assert_eq!(&actual[12..], &input[12..]);
        }

        // Rebinding does not allocate pipelines. Old encoded work keeps its
        // bindings, while both uses of the shared workspace execute in order.
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
        prepared
            .rebind(DenoiseBuffers {
                output: (&output2).into(),
                ..bindings
            })
            .unwrap();
        prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(download(&device, &queue, &output2, bytes), expected);
        assert_eq!(prepared.size(), size);
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: (&color).into(),
                ..bindings
            }),
            Err(Error::InvalidArgument(_))
        ));

        // Tiled execution uses the same transfer and preserves odd-sized
        // image boundaries. Compare against the existing low-level executor.
        let mut tiled_options = DenoiserOptions::default();
        tiled_options.max_tile_size = 16;
        let mut tiled = Denoiser::new(&device, &adapter, &model, tiled_options).unwrap();
        let mut tiled = tiled.prepare(size, bindings).unwrap();
        let config = NetworkConfig {
            kernel: Kernel::ImplicitGemm,
            denoise: Some(DenoiseConfig {
                hdr: true,
                log_transfer: false,
                max_tile_size: 16,
                input_scale: 0.5,
            }),
            ..NetworkConfig::new(size[0], size[1])
        };
        let reference = unsafe {
            Network::with_buffers(
                &device,
                &adapter,
                &model.weights,
                &config,
                &[color.clone(), albedo.clone(), normal.clone()],
                &output2,
                false,
            )
        }
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        tiled.encode(&mut encoder, DenoiseParams { input_scale: 0.5 }).unwrap();
        reference.encode(&queue, &mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            download(&device, &queue, &output, bytes),
            download(&device, &queue, &output2, bytes)
        );

        // Each encode snapshots exposure, including when multiple encodes
        // share one submission. Compare against independently planned scales.
        let half = download(&device, &queue, &output2, bytes);
        let mut twice_config = config.clone();
        twice_config.denoise.as_mut().unwrap().input_scale = 2.0;
        let twice = unsafe {
            Network::with_buffers(
                &device,
                &adapter,
                &model.weights,
                &twice_config,
                &[color.clone(), albedo.clone(), normal.clone()],
                &output2,
                false,
            )
        }
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        twice.encode(&queue, &mut encoder);
        queue.submit([encoder.finish()]);
        let double = download(&device, &queue, &output2, bytes);
        assert_ne!(half, double);
        let snapshot_a = image();
        let snapshot_b = image();
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(matches!(
            tiled.encode(&mut encoder, DenoiseParams { input_scale: f32::NAN }),
            Err(Error::InvalidArgument(_))
        ));
        tiled.encode(&mut encoder, DenoiseParams { input_scale: 0.5 }).unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &snapshot_a, 0, bytes);
        tiled.encode(&mut encoder, DenoiseParams { input_scale: 2.0 }).unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &snapshot_b, 0, bytes);
        queue.submit([encoder.finish()]);
        assert_eq!(download(&device, &queue, &snapshot_a, bytes), half);
        assert_eq!(download(&device, &queue, &snapshot_b, bytes), double);
        assert_eq!(tiled.info().precision, crate::Precision::Fp32);
        assert_eq!(tiled.info().tile_count, 4);
        assert!(tiled.info().kernels.contains(&KernelPath::ImplicitGemm));
        assert!(tiled.info().workspace_bytes > 0 && tiled.info().shared_weight_bytes > 0);

        // A model clone on the same device reuses physical weight allocations,
        // while its prepared workspace remains independent.
        let mut another = Denoiser::new(&device, &adapter, &model.clone(), DenoiserOptions::default()).unwrap();
        let other = another.prepare(size, bindings).unwrap();
        let ids = &prepared.network.metadata.as_ref().unwrap().weights;
        let other_ids = &other.network.metadata.as_ref().unwrap().weights;
        assert!(!ids.is_empty());
        for (&a, &b) in ids.iter().zip(other_ids) {
            assert_eq!(prepared.network.buffers[a], other.network.buffers[b]);
        }
        assert_ne!(
            prepared.network.buffers[prepared.network.metadata.as_ref().unwrap().frame.unwrap()],
            other.network.buffers[other.network.metadata.as_ref().unwrap().frame.unwrap()]
        );
        let limits_error = check_buffer_limits(
            &device,
            "test image",
            device.limits().max_storage_buffer_binding_size + 1,
            true,
        )
        .unwrap_err();
        assert!(
            matches!(limits_error, Error::LimitExceeded { limit: Limit::StorageBindingSize, required, available, .. } if required == available + 1)
        );

        // Host suballocation: inputs share an allocation; output uses another
        // one due to wgpu whole-buffer usage tracking. Both allow offsets.
        let alignment = device.limits().min_storage_buffer_offset_alignment as u64;
        let stride = bytes.div_ceil(alignment) * alignment;
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: stride * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&arena, 0, &data);
        queue.write_buffer(&arena, stride * 2, &normals);
        let output_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: stride * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let arena_bindings = DenoiseBuffers {
            color: ImageBuffer {
                buffer: &arena,
                offset: 0,
            },
            albedo: Some(ImageBuffer {
                buffer: &arena,
                offset: stride,
            }),
            normal: Some(ImageBuffer {
                buffer: &arena,
                offset: stride * 2,
            }),
            output: ImageBuffer {
                buffer: &output_arena,
                offset: stride * 3,
            },
        };
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: ImageBuffer {
                    buffer: &arena,
                    offset: stride * 3
                },
                ..arena_bindings
            }),
            Err(Error::InvalidArgument(_))
        ));
        prepared.rebind(arena_bindings).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
        encoder.copy_buffer_to_buffer(&output_arena, stride * 3, &snapshot_a, 0, bytes);
        queue.submit([encoder.finish()]);
        assert_eq!(download(&device, &queue, &snapshot_a, bytes), expected);
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: ImageBuffer {
                    buffer: &arena,
                    offset: 1
                },
                ..arena_bindings
            }),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: arena_bindings.color,
                ..arena_bindings
            }),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: ImageBuffer {
                    buffer: &arena,
                    offset: u64::MAX
                },
                ..arena_bindings
            }),
            Err(Error::InvalidArgument(_) | Error::BufferTooSmall { .. })
        ));

        if device.limits().max_buffer_size > device.limits().max_storage_buffer_binding_size {
            let large = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: device.limits().max_storage_buffer_binding_size + alignment,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            prepared
                .rebind(DenoiseBuffers {
                    output: ImageBuffer {
                        buffer: &large,
                        offset: alignment,
                    },
                    ..bindings
                })
                .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            prepared.encode(&mut encoder, DenoiseParams::default()).unwrap();
            encoder.copy_buffer_to_buffer(&large, alignment, &snapshot_a, 0, bytes);
            queue.submit([encoder.finish()]);
            assert_eq!(download(&device, &queue, &snapshot_a, bytes), expected);
        }

        // Wrong-device bindings are recoverable errors, not uncaptured wgpu
        // validation panics, and a failed rebind leaves the old bindings intact.
        let (other_device, _) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let foreign = other_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: (&foreign).into(),
                ..bindings
            }),
            Err(Error::Gpu(_))
        ));
        assert!(matches!(
            denoiser.prepare(
                size,
                DenoiseBuffers {
                    output: (&foreign).into(),
                    ..bindings
                }
            ),
            Err(Error::Gpu(_))
        ));

        // A synthetic recorded program writes 1, copies it, writes 2, copies
        // it. Queue writes would incorrectly produce [2, 2] in this case.
        let desc = |size, usage| crate::BufferDescription {
            size,
            usage,
            data: vec![],
            label: String::new(),
        };
        let program = Program {
            metadata: None,
            inputs: vec![],
            output: 1,
            output_external: false,
            width: 1,
            height: 1,
            output_channels: 2,
            buffers: vec![
                desc(4, (wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST).bits()),
                desc(8, (wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST).bits()),
            ],
            shaders: vec![],
            pipelines: vec![],
            bind_groups: vec![],
            operations: vec![
                crate::Operation::Write {
                    buffer: 0,
                    offset: 0,
                    data: 1u32.to_le_bytes().to_vec(),
                },
                crate::Operation::Submit(vec![crate::Command::Copy {
                    source: 0,
                    source_offset: 0,
                    destination: 1,
                    destination_offset: 0,
                    size: 4,
                }]),
                crate::Operation::Write {
                    buffer: 0,
                    offset: 0,
                    data: 2u32.to_le_bytes().to_vec(),
                },
                crate::Operation::Submit(vec![crate::Command::Copy {
                    source: 0,
                    source_offset: 0,
                    destination: 1,
                    destination_offset: 4,
                    size: 4,
                }]),
            ],
        };
        let network = Network::from_program(
            &device,
            program,
            None,
            None,
            false,
            crate::ResourceCaches {
                pipelines: &mut PipelineMap::new(),
                weights: &mut WeightMap::new(),
            },
            0.0,
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        network.encode_commands(&mut encoder);
        network.encode_commands(&mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            download(&device, &queue, network.output_buffer(), 8),
            [1u32.to_le_bytes(), 2u32.to_le_bytes()].concat()
        );
    }
}
