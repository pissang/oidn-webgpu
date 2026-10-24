use crate::{DenoiseConfig, Kernel, Network, NetworkConfig, PipelineMap, Program, Reader};
use std::{fmt, sync::Arc, time::Instant};

/// A failure while loading a model or preparing a denoising.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    InvalidArgument(String),
    Model(String),
    UnsupportedDevice(String),
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, message) = match self {
            Self::InvalidArgument(m) => ("invalid argument", m),
            Self::Model(m) => ("model", m),
            Self::UnsupportedDevice(m) => ("unsupported device", m),
            Self::Gpu(m) => ("GPU", m),
        };
        write!(f, "{kind}: {message}")
    }
}

impl std::error::Error for Error {}

/// Validated OIDN TZA weights, independent of any GPU device.
/// Clones share the owned weights; the original byte slice may be released.
#[derive(Clone, Debug)]
pub struct Model {
    weights: Arc<[u8]>,
    inputs: usize,
}

impl Model {
    pub fn from_tza(weights: &[u8]) -> std::result::Result<Self, Error> {
        let bytes = crate::core_call(|pointer, size| unsafe {
            crate::oidnw_core_inspect_model(weights.as_ptr(), weights.len(), pointer, size);
        })
        .map_err(Error::Model)?;
        let mut reader = Reader {
            bytes: &bytes,
            offset: 0,
        };
        let parsed = (|| -> crate::Result<usize> {
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
                _ => Err("the model must read color, color/albedo, or color/albedo/normal".into()),
            }
        })();
        Ok(Self {
            weights: Arc::from(weights),
            inputs: parsed.map_err(Error::Model)?,
        })
    }

    /// 1 = color, 2 = color and albedo, 3 = color, albedo and normal.
    pub fn input_count(&self) -> usize {
        self.inputs
    }
}

/// HDR transfer expected by the model's training.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transfer {
    /// Regular RT HDR models.
    #[default]
    Pu,
    /// RTLightmap HDR models.
    Log,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DenoiserOptions {
    /// True for linear HDR color. LDR also uses linear f32 RGB in [0, 1].
    pub hdr: bool,
    pub transfer: Transfer,
    /// Maximum output tile edge, rounded down to a multiple of 16; 0 uses
    /// one tile. Tiling reduces working memory, but not the image buffers.
    pub max_tile_size: u32,
    /// Positive finite HDR scale, applied before transfer and undone on output.
    /// This API does not estimate exposure automatically.
    pub input_scale: f32,
}

impl Default for DenoiserOptions {
    fn default() -> Self {
        Self {
            hdr: true,
            transfer: Transfer::Pu,
            max_tile_size: 0,
            input_scale: 1.0,
        }
    }
}

/// Host-owned, unmapped buffers from the denoiser's device. Each must have
/// STORAGE usage and hold at least width * height * 16 bytes from offset zero.
/// All supplied buffers must be distinct. Extra auxiliary images are rejected.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseBuffers<'a> {
    pub color: &'a wgpu::Buffer,
    /// Linear albedo RGB in [0, 1]; alpha ignored.
    pub albedo: Option<&'a wgpu::Buffer>,
    /// Shading normal RGB in [-1, 1], in world or view space. Accumulated
    /// normals need not have unit length; no remapping is applied. Alpha ignored.
    pub normal: Option<&'a wgpu::Buffer>,
    /// RGBA f32 output, with the color input's alpha preserved.
    pub output: &'a wgpu::Buffer,
}

impl DenoiseBuffers<'_> {
    fn validate(&self, count: usize, bytes: u64) -> std::result::Result<Vec<wgpu::Buffer>, Error> {
        if self.albedo.is_some() != (count >= 2) || self.normal.is_some() != (count >= 3) {
            return Err(Error::InvalidArgument(format!(
                "model requires {count} images: color, then albedo, then normal; provide exactly those inputs"
            )));
        }
        let inputs: Vec<_> = [Some(self.color), self.albedo, self.normal]
            .into_iter()
            .flatten()
            .collect();
        let mut all = inputs.clone();
        all.push(self.output);
        for (i, buffer) in all.iter().enumerate() {
            if !buffer.usage().contains(wgpu::BufferUsages::STORAGE) || buffer.size() < bytes {
                return Err(Error::InvalidArgument(format!(
                    "image buffer {i} must allow STORAGE and hold at least {bytes} bytes"
                )));
            }
            if all[..i].contains(buffer) {
                return Err(Error::InvalidArgument("image buffers must not alias".into()));
            }
        }
        Ok(inputs.into_iter().cloned().collect())
    }
}

/// Device-bound model and pipeline cache. Construction does not compile
/// pipelines; call prepare before the first interactive frame.
pub struct Denoiser {
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    model: Model,
    options: DenoiserOptions,
    pipelines: PipelineMap,
    unchecked: bool,
}

impl Denoiser {
    /// Baseline FP32 denoising needs no optional features.
    pub fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    /// Optional stable features to request before creating the host device.
    /// Cooperative matrices may additionally be enabled by hosts accepting
    /// wgpu's experimental feature contract; otherwise auto uses implicit GEMM.
    pub fn recommended_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & (wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP)
    }

    /// Baseline limits to request when creating the device. Larger images
    /// may require higher buffer limits; prepare reports insufficient limits.
    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }

    /// The adapter must be the one that created device. Shader runtime checks
    /// and workgroup zero-initialization are enabled on this safe path.
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        model: &Model,
        options: DenoiserOptions,
    ) -> std::result::Result<Self, Error> {
        if device.adapter_info() != adapter.get_info() {
            return Err(Error::InvalidArgument("adapter must match device".into()));
        }
        if !(options.input_scale.is_finite() && options.input_scale > 0.0) {
            return Err(Error::InvalidArgument("input_scale must be positive and finite".into()));
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
            unchecked: false,
        })
    }

    /// Like new, but disables shader bounds checks and workgroup initialization.
    ///
    /// # Safety
    /// The caller must ensure generated kernels stay in bounds and initialize
    /// workgroup data before reading it for every prepared configuration.
    /// Violations can read or write arbitrary GPU memory.
    pub unsafe fn new_unchecked(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        model: &Model,
        options: DenoiserOptions,
    ) -> std::result::Result<Self, Error> {
        let mut denoiser = Self::new(device, adapter, model, options)?;
        denoiser.unchecked = true;
        Ok(denoiser)
    }

    /// Plans a fixed image size and allocates its workspaces. May block while
    /// shaders compile. Pipelines are reused across sizes on this denoiser.
    /// Each returned instance owns a separate workspace and keeps buffer handles
    /// alive; do not explicitly destroy those buffers while work is pending.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
        buffers: DenoiseBuffers<'_>,
    ) -> std::result::Result<PreparedDenoise, Error> {
        if device != &self.device {
            return Err(Error::InvalidArgument("prepare requires the denoiser's device".into()));
        }
        let [width, height] = size;
        if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
            return Err(Error::InvalidArgument("width and height must be in 1..=16384".into()));
        }
        let bytes = width as u64 * height as u64 * 16;
        if bytes > device.limits().max_buffer_size
            || bytes > device.limits().max_storage_buffer_binding_size
            || bytes > u32::MAX as u64
        {
            return Err(Error::UnsupportedDevice(
                "image exceeds the device buffer/binding limits; tiling does not shrink external images".into(),
            ));
        }
        let inputs = buffers.validate(self.model.inputs, bytes)?;
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
                input_scale: self.options.input_scale,
            }),
            ..NetworkConfig::new(width, height)
        };
        let started = Instant::now();
        let program = crate::plan(device, &self.adapter, &self.model.weights, &config).map_err(Error::Model)?;
        check_limits(device, &program)?;
        let planning_ms = started.elapsed().as_secs_f64() * 1e3;
        let network = gpu_checked(device, || {
            Network::from_program(
                device,
                program,
                Some(&inputs),
                Some(buffers.output),
                self.unchecked,
                &mut self.pipelines,
                planning_ms,
            )
            .map_err(Error::Model)
        });
        match network {
            Ok(network) => Ok(PreparedDenoise {
                device: device.clone(),
                network,
                inputs: self.model.inputs,
                image_bytes: bytes,
            }),
            Err(error) => {
                self.pipelines.clear();
                Err(error)
            }
        }
    }
}

fn check_limits(device: &wgpu::Device, program: &Program) -> std::result::Result<(), Error> {
    let limits = device.limits();
    for buffer in &program.buffers {
        if buffer.size > limits.max_buffer_size
            || (buffer.usage & wgpu::BufferUsages::STORAGE.bits() != 0
                && buffer.size > limits.max_storage_buffer_binding_size)
        {
            return Err(Error::UnsupportedDevice(format!(
                "buffer {} needs {} bytes; device buffer/binding limits are {}/{}; request higher limits or use smaller tiles/images",
                buffer.label, buffer.size, limits.max_buffer_size, limits.max_storage_buffer_binding_size
            )));
        }
    }
    Ok(())
}

fn gpu_checked<T>(
    device: &wgpu::Device,
    body: impl FnOnce() -> std::result::Result<T, Error>,
) -> std::result::Result<T, Error> {
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

/// Prepared fixed-size denoising with its own workspace. This object may outlive
/// its Denoiser. Submit its executions in encode order on the device's queue.
/// For independent workspaces, prepare separate instances.
pub struct PreparedDenoise {
    device: wgpu::Device,
    network: Network,
    inputs: usize,
    image_bytes: u64,
}

impl PreparedDenoise {
    pub fn size(&self) -> [u32; 2] {
        [self.network.width, self.network.height]
    }

    /// Appends GPU work only: no queue writes, submission, CPU wait or image
    /// readback. The encoder must belong to this instance's device. Complete
    /// any active render/compute pass before calling this method.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) -> std::result::Result<(), Error> {
        self.network.encode_commands(encoder);
        Ok(())
    }

    /// Replaces image bindings at the same size without replanning or creating
    /// pipelines/workspaces. Previously encoded commands retain old bindings.
    /// New buffers must belong to the same device and satisfy DenoiseBuffers.
    pub fn rebind(&mut self, buffers: DenoiseBuffers<'_>) -> std::result::Result<(), Error> {
        let inputs = buffers.validate(self.inputs, self.image_bytes)?;
        let mut candidate = self.network.buffers.clone();
        for (&id, buffer) in self.network.inputs.iter().zip(inputs) {
            candidate[id] = buffer;
        }
        candidate[self.network.output] = buffers.output.clone();
        let bind_groups = gpu_checked(&self.device, || {
            Ok(self
                .network
                .bind_group_descriptions
                .iter()
                .map(|group| {
                    let entries: Vec<_> = group
                        .entries
                        .iter()
                        .map(|&[binding, buffer, offset, size]| wgpu::BindGroupEntry {
                            binding,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &candidate[buffer as usize],
                                offset: offset as u64,
                                size: std::num::NonZeroU64::new(size as u64),
                            }),
                        })
                        .collect();
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("oidn-webgpu rebound images"),
                        layout: &self.network.pipelines[group.pipeline].get_bind_group_layout(group.group),
                        entries: &entries,
                    })
                })
                .collect())
        })?;
        self.network.buffers = candidate;
        self.network.bind_groups = bind_groups;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            color: &color,
            albedo: Some(&albedo),
            normal: Some(&normal),
            output: &output,
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
            denoiser.prepare(&device, [0, 19], bindings),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            denoiser.prepare(
                &device,
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
                &device,
                size,
                DenoiseBuffers {
                    output: &color,
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
                &device,
                size,
                DenoiseBuffers {
                    output: &short,
                    ..bindings
                }
            ),
            Err(Error::InvalidArgument(_))
        ));
        let mut bad = DenoiserOptions::default();
        bad.input_scale = f32::NAN;
        assert!(matches!(
            Denoiser::new(&device, &adapter, &model, bad),
            Err(Error::InvalidArgument(_))
        ));

        let mut prepared = denoiser.prepare(&device, size, bindings).unwrap();
        let pipelines = denoiser.pipelines.clone();
        let _other_size = denoiser.prepare(&device, [16, 16], bindings).unwrap();
        for (key, pipeline) in &pipelines {
            assert_eq!(denoiser.pipelines.get(key), Some(pipeline));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        prepared.encode(&mut encoder).unwrap();
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
        prepared.encode(&mut encoder).unwrap();
        prepared
            .rebind(DenoiseBuffers {
                output: &output2,
                ..bindings
            })
            .unwrap();
        prepared.encode(&mut encoder).unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(download(&device, &queue, &output2, bytes), expected);
        assert_eq!(prepared.size(), size);
        assert!(matches!(
            prepared.rebind(DenoiseBuffers {
                output: &color,
                ..bindings
            }),
            Err(Error::InvalidArgument(_))
        ));

        // Tiled execution uses the same transfer and preserves odd-sized
        // image boundaries. Compare against the existing low-level executor.
        let mut tiled_options = DenoiserOptions::default();
        tiled_options.max_tile_size = 16;
        tiled_options.input_scale = 0.5;
        let mut tiled = Denoiser::new(&device, &adapter, &model, tiled_options).unwrap();
        let mut tiled = tiled.prepare(&device, size, bindings).unwrap();
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
        tiled.encode(&mut encoder).unwrap();
        reference.encode(&queue, &mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            download(&device, &queue, &output, bytes),
            download(&device, &queue, &output2, bytes)
        );

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
                output: &foreign,
                ..bindings
            }),
            Err(Error::Gpu(_))
        ));
        assert!(matches!(
            denoiser.prepare(
                &device,
                size,
                DenoiseBuffers {
                    output: &foreign,
                    ..bindings
                }
            ),
            Err(Error::Gpu(_))
        ));
        assert!(matches!(
            denoiser.prepare(&other_device, size, bindings),
            Err(Error::InvalidArgument(_))
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
        let network = Network::from_program(&device, program, None, None, false, &mut PipelineMap::new(), 0.0).unwrap();
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
