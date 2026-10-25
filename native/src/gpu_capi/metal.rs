use super::*;

#[cfg(not(target_os = "macos"))]
pub(super) unsafe fn import(_: &MetalDeviceDesc) -> Outcome<Context> {
    Err(unavailable())
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::{
        MTLBuffer, MTLCommandQueue, MTLDevice, MTLEvent, MTLResource, MTLSharedEvent,
        MTLStorageMode,
    };

    struct Metal {
        registry: u64,
    }
    struct Event(Retained<ProtocolObject<dyn MTLSharedEvent>>);

    pub(super) unsafe fn import(desc: &MetalDeviceDesc) -> Outcome<Context> {
        let raw = unsafe { Retained::retain(desc.device.cast::<ProtocolObject<dyn MTLDevice>>()) }
            .ok_or_else(|| invalid("Metal device is NULL"))?;
        let queue = if desc.queue.is_null() {
            raw.newCommandQueueWithMaxCommandBufferCount(4096)
                .ok_or_else(|| gpu("Metal queue allocation failed"))?
        } else {
            unsafe { Retained::retain(desc.queue.cast::<ProtocolObject<dyn MTLCommandQueue>>()) }
                .ok_or_else(|| invalid("Metal queue is NULL"))?
        };
        if queue.device().registryID() != raw.registryID() {
            return Err(invalid("queue belongs to another Metal device"));
        }
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL))
            .into_iter()
            .find(|adapter| unsafe {
                adapter
                    .as_hal::<wgpu::hal::api::Metal>()
                    .is_some_and(|h| h.raw_device().registryID() == raw.registryID())
            })
            .ok_or_else(|| {
                Failure(
                    OIDNW_ERROR_UNAVAILABLE,
                    "host Metal device has no matching adapter".into(),
                )
            })?;
        let mut features = Denoiser::recommended_features(&adapter);
        if desc.flags & 1 != 0 {
            features.remove(wgpu::Features::SHADER_F16);
        }
        if desc.flags & 2 != 0 {
            features.remove(wgpu::Features::SUBGROUP);
        }
        let limits = adapter.limits();
        let hal_device =
            unsafe { wgpu::hal::metal::Device::device_from_raw(raw.clone(), features, &limits) };
        let hal_queue = unsafe { wgpu::hal::metal::Queue::queue_from_raw(queue, 1.0) };
        hal_queue
            .enable_strict_event_sync()
            .map_err(|e| gpu(e.to_string()))?;
        let (device, queue) = unsafe {
            adapter.create_device_from_hal::<wgpu::hal::api::Metal>(
                wgpu::hal::OpenDevice {
                    device: hal_device,
                    queue: hal_queue,
                },
                &wgpu::DeviceDescriptor {
                    label: Some("oidn GPU C API imported Metal device"),
                    required_features: features,
                    required_limits: limits,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| Failure(OIDNW_ERROR_UNAVAILABLE, e.to_string()))?;
        Ok(Context {
            adapter,
            device,
            queue,
            backend: Box::new(Metal {
                registry: raw.registryID(),
            }),
        })
    }
    impl Backend for Metal {
        unsafe fn buffer(
            &self,
            device: &wgpu::Device,
            raw: *mut c_void,
            size: u64,
        ) -> Outcome<wgpu::Buffer> {
            let raw = unsafe { Retained::retain(raw.cast::<ProtocolObject<dyn MTLBuffer>>()) }
                .ok_or_else(|| invalid("buffer is NULL"))?;
            if raw.device().registryID() != self.registry
                || size != raw.length() as u64
                || size > device.limits().max_buffer_size
                || raw.storageMode() == MTLStorageMode::Memoryless
            {
                return Err(invalid(
                    "Metal buffer device, length or storage mode is incompatible",
                ));
            }
            crate::denoiser::gpu_checked(device, || {
                Ok(unsafe {
                    device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                        wgpu::hal::metal::Device::buffer_from_raw(raw, size),
                        &wgpu::BufferDescriptor {
                            label: Some("oidn C API host Metal buffer"),
                            size,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        },
                    )
                })
            })
            .map_err(denoise_error)
        }
        unsafe fn timeline(&self, raw: *mut c_void) -> Outcome<Box<dyn Timeline>> {
            let raw = unsafe { Retained::retain(raw.cast::<ProtocolObject<dyn MTLSharedEvent>>()) }
                .ok_or_else(|| invalid("event is NULL"))?;
            // MTLSharedEvent can have no associated device (unlike MTLEvent),
            // because it supports CPU and cross-device synchronization.
            if raw
                .device()
                .is_some_and(|device| device.registryID() != self.registry)
            {
                return Err(invalid("event belongs to another Metal device"));
            }
            Ok(Box::new(Event(raw)))
        }
    }
    impl Timeline for Event {
        fn value(&self) -> Outcome<u64> {
            Ok(self.0.signaledValue())
        }
        fn stage_wait(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
            let hal = unsafe { queue.as_hal::<wgpu::hal::api::Metal>() }.ok_or_else(unavailable)?;
            hal.add_wait_event(self.0.clone(), value);
            Ok(())
        }
        fn stage_signal(&self, queue: &wgpu::Queue, value: u64) -> Outcome<()> {
            let hal = unsafe { queue.as_hal::<wgpu::hal::api::Metal>() }.ok_or_else(unavailable)?;
            hal.add_signal_event(self.0.clone(), value);
            Ok(())
        }
        fn unstage(&self, queue: &wgpu::Queue) {
            if let Some(hal) = unsafe { queue.as_hal::<wgpu::hal::api::Metal>() } {
                hal.remove_wait_event(&self.0);
                hal.remove_signal_event(&self.0);
            }
        }
    }
}
#[cfg(target_os = "macos")]
pub(super) unsafe fn import(desc: &MetalDeviceDesc) -> Outcome<Context> {
    unsafe { platform::import(desc) }
}
