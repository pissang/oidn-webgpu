#[path = "common.rs"]
mod common;
#[path = "../../src/gpu_capi/vulkan_features.rs"]
mod features;
#[path = "gpu_client.rs"]
mod gpu_client;
#[path = "queue_lock.rs"]
mod queue_lock;
use ash::{vk, vk::Handle};
use common::*;
use oidn_webgpu::gpu_capi::*;
use oidn_webgpu::{DenoiseParams, Denoiser, DenoiserOptions, Model, wgpu};
use std::{
    collections::{HashSet, VecDeque},
    ffi::{CStr, c_void},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

type Api = wgpu::hal::api::Vulkan;
const TIMEOUT: u64 = 60_000_000_000;
static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);
static NAGA_LAYOUT_ERRORS: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn debug(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    // The pinned HAL suppresses this Naga issue in its own messenger (#7696).
    // Keep it visible and count it separately; never claim a clean validation
    // run. Match only the observed workgroup ArrayStride diagnostic.
    if unsafe { (*data).message_id_number } == 0xb210f7c2_u32 as i32
        && message.contains("Workgroup")
        && message.contains("ArrayStride")
    {
        NAGA_LAYOUT_ERRORS.fetch_add(1, Ordering::Relaxed);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
    }
    eprintln!("Vulkan {severity:?}: {message}");
    vk::FALSE
}

// Host RAII remains responsible for VkInstance/VkDevice destruction. HAL's
// callbacks retain this owner and never destroy these objects themselves.
struct InstanceOwner {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}
impl Drop for InstanceOwner {
    fn drop(&mut self) {
        unsafe {
            self.debug
                .destroy_debug_utils_messenger(self.messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
struct Host {
    owner: Arc<InstanceOwner>,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    family: u32,
    queues: [vk::Queue; 3],
    lock: Arc<queue_lock::QueueLock>,
}
impl std::ops::Deref for Host {
    type Target = InstanceOwner;
    fn deref(&self) -> &InstanceOwner {
        &self.owner
    }
}
impl Drop for Host {
    fn drop(&mut self) {
        unsafe {
            let _guard = self.lock.lock().unwrap();
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    c_device: *mut GpuDevice,
    lock: Arc<queue_lock::QueueLock>,
}
impl Context {
    fn wait(&self, index: wgpu::SubmissionIndex) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(60)),
        })?;
        Ok(())
    }
}

unsafe fn setup(options: &Options) -> Result<(Arc<Host>, Context, serde_json::Value)> {
    unsafe {
        let entry = ash::Entry::load()?;
        let flags = wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG;
        let api = vk::API_VERSION_1_3;
        let extensions = wgpu::hal::vulkan::Instance::desired_extensions(&entry, api, flags)?;
        let extension_names: Vec<_> = extensions.iter().map(|e| e.as_ptr()).collect();
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let app = vk::ApplicationInfo::default()
            .api_version(api)
            .application_name(c"oidn native interop host");
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&extension_names)
                .enabled_layer_names(&layers),
            None,
        )?;
        let debug_ext = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let messenger = debug_ext.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug)),
            None,
        )?;
        let owner = Arc::new(InstanceOwner {
            _entry: entry.clone(),
            instance: instance.clone(),
            debug: debug_ext,
            messenger,
        });
        let physical = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find(|p| {
                CStr::from_ptr(
                    instance
                        .get_physical_device_properties(*p)
                        .device_name
                        .as_ptr(),
                )
                .to_string_lossy()
                .contains(&options.adapter)
            })
            .ok_or("host physical device not found")?;
        let props = instance.get_physical_device_properties(physical);
        // Route B feasibility query only: no second device is substituted for the
        // host device import below. Export/import execution is a separate prototype.
        let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let mut external_memory = vk::ExternalBufferProperties::default();
        instance.get_physical_device_external_buffer_properties(
            physical,
            &external_info,
            &mut external_memory,
        );
        let mut timeline_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let external_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32)
            .push_next(&mut timeline_type);
        let mut external_sync = vk::ExternalSemaphoreProperties::default();
        instance.get_physical_device_external_semaphore_properties(
            physical,
            &external_info,
            &mut external_sync,
        );
        let external_probe = serde_json::json!({
            "handle_type":"OPAQUE_WIN32", "execution_verified":false,
            "buffer_features":format!("{:?}", external_memory.external_memory_properties.external_memory_features),
            "compatible_memory_handles":format!("{:?}", external_memory.external_memory_properties.compatible_handle_types),
            "timeline_features":format!("{:?}", external_sync.external_semaphore_features),
            "compatible_semaphore_handles":format!("{:?}", external_sync.compatible_handle_types),
        });
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        instance.get_physical_device_properties2(
            physical,
            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut ids),
        );
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|q| {
                q.queue_count >= 3
                    && q.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .ok_or("prototype requires three queues in one graphics/compute family")?
            as u32;
        // Reuse the HOST loader table, including any host interception/layers.
        let keep = owner.clone();
        let hal_instance = wgpu::hal::vulkan::Instance::from_raw(
            entry.clone(),
            instance.clone(),
            api,
            0,
            None,
            extensions,
            flags,
            Default::default(),
            false,
            Some(Box::new(move || drop(keep))),
        )?;
        let exposed = hal_instance
            .expose_adapter(physical)
            .ok_or("HAL cannot expose host physical device")?;
        let wg_instance = wgpu::Instance::from_hal::<Api>(hal_instance);
        let adapter = wg_instance.create_adapter_from_hal(exposed);
        let mut requested = if options.recommended {
            Denoiser::recommended_features(&adapter)
        } else {
            Denoiser::required_features()
        };
        if options.fp32 {
            requested.remove(wgpu::Features::SHADER_F16);
        }
        let mut limits = adapter.limits();
        // wgpu-core's indirect validator uses 32-bit buffer addresses even on
        // adapters reporting a larger maxBufferSize.
        limits.max_buffer_size = limits.max_buffer_size.min(u32::MAX as u64);
        let hal = adapter.as_hal::<Api>().ok_or("not Vulkan")?;
        let device_extensions = hal.required_device_extensions(requested);
        let mut requirements = hal.physical_device_features(&device_extensions, requested);
        // This query and snapshot happen BEFORE vkCreateDevice. Snapshot every HAL
        // boolean, including capabilities implicitly used outside denoise shaders.
        let priorities = [1.0, 1.0, 1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let names: Vec<_> = device_extensions.iter().map(|e| e.as_ptr()).collect();
        let create = requirements.add_to_device_create(
            vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&names),
        );
        let required = features::snapshot(&create)?;
        let actual = features::snapshot(&create)?;
        features::validate_create(
            &required,
            &device_extensions,
            &create,
            family,
            if options.shared { 0 } else { 1 },
        )?;
        if actual
            .get("PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES")
            .and_then(|v| v.first())
            != Some(&1)
        {
            return Err("prototype requires timelineSemaphore; adapter does not enable it".into());
        }
        // Reject a host declaration missing each individual required HAL feature.
        let negative_checks = features::negative_checks(&required)?;
        let device = instance.create_device(physical, &create, None)?;
        let raw_queues = [
            device.get_device_queue(family, 0),
            device.get_device_queue(family, 1),
            device.get_device_queue(family, 2),
        ];
        let host = Arc::new(Host {
            owner,
            device,
            physical,
            family,
            queues: raw_queues,
            lock: Arc::new(queue_lock::QueueLock::default()),
        });
        // Instance's non-owning wrapper must also keep the host alive. The adapter
        // above was only used to query requirements; wrap a second non-owning view.
        drop(hal);
        drop(adapter);
        drop(wg_instance);
        let keep = host.clone();
        let hal_instance = wgpu::hal::vulkan::Instance::from_raw(
            host._entry.clone(),
            host.instance.clone(),
            api,
            0,
            None,
            wgpu::hal::vulkan::Instance::desired_extensions(&host._entry, api, flags)?,
            flags,
            Default::default(),
            false,
            Some(Box::new(move || drop(keep))),
        )?;
        let exposed = hal_instance
            .expose_adapter(physical)
            .ok_or("missing physical device")?;
        let wg_instance = wgpu::Instance::from_hal::<Api>(hal_instance);
        let adapter = wg_instance.create_adapter_from_hal(exposed);
        let keep = host.clone();
        let _guard = host.lock.lock().unwrap();
        let hal = adapter.as_hal::<Api>().unwrap();
        // DeviceCreateInfo is the original host declaration, not a post-creation
        // physical-device capability query. Aggregated 1.1/1.2/1.3 feature structs
        // are intentionally not normalized by this prototype and are rejected.
        features::validate_create(
            &required,
            &device_extensions,
            &create,
            family,
            if options.shared { 0 } else { 1 },
        )?;
        let open = hal.device_from_raw(
            host.device.clone(),
            Some(Box::new(move || drop(keep))),
            &device_extensions,
            requested,
            &limits,
            &Default::default(),
            family,
            if options.shared { 0 } else { 1 },
        )?;
        drop(hal);
        let (device, queue) = adapter.create_device_from_hal::<Api>(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("imported HOST VkDevice"),
                required_features: requested,
                required_limits: limits,
                ..Default::default()
            },
        )?;
        drop(_guard);
        let mut c_device = std::ptr::null_mut();
        if options.c_api {
            let host_instance_create = vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&extension_names)
                .enabled_layer_names(&layers);
            let desc = VulkanDeviceDesc {
                flags: u32::from(options.fp32),
                get_instance_proc_addr: Some(host._entry.static_fn().get_instance_proc_addr),
                instance: host.instance.handle(),
                physical_device: physical,
                device: host.device.handle(),
                queue: host.queues[if options.shared { 0 } else { 1 }],
                queue_family_index: family,
                queue_index: if options.shared { 0 } else { 1 },
                instance_create_info: (&host_instance_create as *const vk::InstanceCreateInfo)
                    .cast(),
                device_create_info: (&create as *const vk::DeviceCreateInfo).cast(),
                lock: Some(queue_lock::lock),
                unlock: Some(queue_lock::unlock),
                user_data: Arc::as_ptr(&host.lock).cast_mut().cast(),
                max_pending_submissions: 8,
            };
            let mut error = [0i8; 4096];
            if oidnw_gpu_device_import_vulkan(&desc, &mut c_device, error.as_mut_ptr(), error.len())
                != 0
            {
                return Err(CStr::from_ptr(error.as_ptr())
                    .to_string_lossy()
                    .into_owned()
                    .into());
            }
        }
        let report = serde_json::json!({"device_uuid": ids.device_uuid, "driver_uuid": ids.driver_uuid,
        "driver_version": props.driver_version, "external_memory_route_probe":external_probe,
        "required_extensions": device_extensions.iter().map(|s| s.to_string_lossy()).collect::<Vec<_>>(),
        "required_hal_features": required, "missing_feature_rejections": negative_checks,
        "host_device": host.device.handle().as_raw(), "imported_device": device.as_hal::<Api>().unwrap().raw_device().handle().as_raw()});
        let lock = host.lock.clone();
        Ok((
            host,
            Context {
                device,
                queue,
                adapter,
                c_device,
                lock,
            },
            report,
        ))
    }
}

struct Buffer {
    host: Arc<Host>,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.host.device.destroy_buffer(self.raw, None);
            self.host.device.free_memory(self.memory, None);
        }
    }
}
impl Buffer {
    unsafe fn new(host: &Arc<Host>, size: u64, cpu: bool) -> Result<Arc<Self>> {
        unsafe {
            let raw = host.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = host.device.get_buffer_memory_requirements(raw);
            let props = host
                .instance
                .get_physical_device_memory_properties(host.physical);
            let flags = if cpu {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            let index = props
                .memory_types_as_slice()
                .iter()
                .enumerate()
                .find(|(i, p)| {
                    req.memory_type_bits & (1 << i) != 0 && p.property_flags.contains(flags)
                })
                .ok_or("no memory type")?
                .0;
            let memory = host.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index as u32),
                None,
            )?;
            host.device.bind_buffer_memory(raw, memory, 0)?;
            Ok(Arc::new(Self {
                host: host.clone(),
                raw,
                memory,
                size,
            }))
        }
    }
}
#[derive(Default)]
struct Imports {
    live: Arc<Mutex<HashSet<u64>>>,
    drops: Arc<AtomicUsize>,
}
impl Imports {
    unsafe fn import(&self, context: &Context, raw: &Arc<Buffer>) -> Result<wgpu::Buffer> {
        unsafe {
            if context
                .device
                .as_hal::<Api>()
                .unwrap()
                .raw_device()
                .handle()
                != raw.host.device.handle()
            {
                return Err("buffer device mismatch".into());
            }
            let id = raw.raw.as_raw();
            if !self.live.lock().unwrap().insert(id) {
                return Err("duplicate native buffer import".into());
            }
            let keep = raw.clone();
            let live = self.live.clone();
            let drops = self.drops.clone();
            let hal = wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                raw.raw,
                Box::new(move || {
                    live.lock().unwrap().remove(&id);
                    drops.fetch_add(1, Ordering::Relaxed);
                    drop(keep);
                }),
            );
            Ok(context.device.create_buffer_from_hal::<Api>(
                hal,
                &wgpu::BufferDescriptor {
                    label: Some("host-owned VkBuffer"),
                    size: raw.size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            ))
        }
    }
}
struct Kernels {
    host: Arc<Host>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pipelines: [vk::Pipeline; 2],
}
impl Kernels {
    unsafe fn new(host: &Arc<Host>) -> Result<Self> {
        unsafe {
            let code = [spirv("produce")?, spirv("consume")?];
            let bindings: Vec<_> = (0..3)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = host.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sets = [set_layout];
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(16)];
            let layout = host.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sets)
                    .push_constant_ranges(&push),
                None,
            )?;
            let mut pipelines = Vec::new();
            for (entry, code) in ["produce", "consume"].into_iter().zip(&code) {
                let module = host.device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code),
                    None,
                )?;
                let name = std::ffi::CString::new(entry)?;
                let info = vk::ComputePipelineCreateInfo::default()
                    .layout(layout)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(&name),
                    );
                let result = host
                    .device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e);
                host.device.destroy_shader_module(module, None);
                pipelines.push(result?[0]);
            }
            Ok(Self {
                host: host.clone(),
                layout,
                set_layout,
                pipelines: pipelines.try_into().unwrap(),
            })
        }
    }
}
impl Drop for Kernels {
    fn drop(&mut self) {
        unsafe {
            for p in self.pipelines {
                self.host.device.destroy_pipeline(p, None);
            }
            self.host.device.destroy_pipeline_layout(self.layout, None);
            self.host
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
struct Batch {
    host: Arc<Host>,
    pool: vk::CommandPool,
    descriptors: vk::DescriptorPool,
    command: vk::CommandBuffer,
}
impl Batch {
    unsafe fn new(
        k: &Kernels,
        consume: bool,
        buffers: [(&Buffer, u64, u64); 3],
        params: [u32; 4],
    ) -> Result<Self> {
        unsafe {
            let host = &k.host;
            let pool = host.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(host.family),
                None,
            )?;
            let command = host.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)];
            let descriptors = host.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [k.set_layout];
            let set = host.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptors)
                    .set_layouts(&layouts),
            )?[0];
            for (i, (b, offset, range)) in buffers.iter().enumerate() {
                let info = [vk::DescriptorBufferInfo::default()
                    .buffer(b.raw)
                    .offset(*offset)
                    .range(*range)];
                host.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
            }
            host.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            host.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                k.pipelines[consume as usize],
            );
            host.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                k.layout,
                0,
                &[set],
                &[],
            );
            let bytes: Vec<_> = params.iter().flat_map(|p| p.to_le_bytes()).collect();
            host.device.cmd_push_constants(
                command,
                k.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            host.device
                .cmd_dispatch(command, (params[0] * params[0]).div_ceil(64), 1, 1);
            if consume {
                // Timeline completion is an execution dependency. Make the
                // validation writes available to the HOST domain before the
                // CPU reads this HOST_COHERENT allocation after its wait.
                let barriers = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                host.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barriers,
                    &[],
                    &[],
                );
            }
            host.device.end_command_buffer(command)?;
            Ok(Self {
                host: host.clone(),
                pool,
                descriptors,
                command,
            })
        }
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            self.host.device.destroy_command_pool(self.pool, None);
            self.host
                .device
                .destroy_descriptor_pool(self.descriptors, None);
        }
    }
}
struct Timelines {
    host: Arc<Host>,
    sems: [vk::Semaphore; 7],
}
impl Timelines {
    unsafe fn new(host: &Arc<Host>) -> Result<Self> {
        unsafe {
            let mut sems = Vec::new();
            for _ in 0..7 {
                let mut ty = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE);
                sems.push(host.device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut ty),
                    None,
                )?);
            }
            Ok(Self {
                host: host.clone(),
                sems: sems.try_into().unwrap(),
            })
        }
    }
    fn value(&self, i: usize) -> Result<u64> {
        Ok(unsafe { self.host.device.get_semaphore_counter_value(self.sems[i])? })
    }
    fn signal(&self, i: usize) -> Result<()> {
        unsafe {
            self.host.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.sems[i])
                    .value(1),
            )?;
        }
        Ok(())
    }
    fn wait(&self, i: usize, value: u64, timeout: u64) -> Result<()> {
        unsafe {
            self.host.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.sems[i]])
                    .values(&[value]),
                timeout,
            )?;
        }
        Ok(())
    }
}
impl Drop for Timelines {
    fn drop(&mut self) {
        unsafe {
            for s in self.sems {
                self.host.device.destroy_semaphore(s, None);
            }
        }
    }
}
unsafe fn submit(
    host: &Host,
    queue: vk::Queue,
    batch: Option<&Batch>,
    waits: &[(vk::Semaphore, u64)],
    signals: &[(vk::Semaphore, u64)],
) -> Result<()> {
    unsafe {
        let _guard = host.lock.lock().unwrap();
        let wait_sems: Vec<_> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<_> = waits.iter().map(|w| w.1).collect();
        let signal_sems: Vec<_> = signals.iter().map(|w| w.0).collect();
        let signal_values: Vec<_> = signals.iter().map(|w| w.1).collect();
        let stages = vec![vk::PipelineStageFlags::TOP_OF_PIPE; waits.len()];
        let commands: Vec<_> = batch.map(|b| b.command).into_iter().collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        host.device.queue_submit(
            queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&stages)
                .signal_semaphores(&signal_sems)
                .command_buffers(&commands)
                .push_next(&mut timeline)],
            vk::Fence::null(),
        )?;
        Ok(())
    }
}
unsafe fn library_submit(
    context: &Context,
    host: &Host,
    command: Option<wgpu::CommandBuffer>,
    waits: &[(vk::Semaphore, u64)],
    signal: (vk::Semaphore, u64),
) -> Result<wgpu::SubmissionIndex> {
    unsafe {
        let _guard = host.lock.lock().unwrap();
        // Stage external synchronization only after all argument checks/encoding.
        let hal = context.queue.as_hal::<Api>().unwrap();
        for &(s, v) in waits {
            hal.add_wait_semaphore(s, Some(v), vk::PipelineStageFlags::TOP_OF_PIPE);
        }
        hal.add_signal_semaphore(signal.0, Some(signal.1));
        drop(hal);
        Ok(context.queue.submit(command))
    }
}

pub fn run() -> Result<()> {
    unsafe {
        let options = Options::parse()?;
        if !options.debug {
            return Err("Vulkan prototype requires validation; --no-debug is DX12-only".into());
        }
        let (host, context, requirements) = setup(&options)?;
        assert_eq!(requirements["host_device"], requirements["imported_device"]);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let capture = errors.clone();
        context.device.on_uncaptured_error(Arc::new(move |e| {
            eprintln!("unrecoverable wgpu error; submission state is uncertain: {e}");
            capture.lock().unwrap().push(e.to_string());
            // Test-process policy only: do not return possibly in-flight native
            // allocations to a host or resume filtering after uncertain submit.
            std::process::exit(2);
        }));
        let capture = errors.clone();
        context
            .device
            .set_device_lost_callback(move |reason, message| {
                if reason != wgpu::DeviceLostReason::Destroyed {
                    eprintln!(
                        "device lost; refusing to reuse in-flight state: {reason:?} {message}"
                    );
                    capture
                        .lock()
                        .unwrap()
                        .push(format!("device lost: {reason:?} {message}"));
                    std::process::exit(2);
                }
            });
        let bytes = options.size as u64 * options.size as u64 * 16;
        let report_bytes = options.frames as u64 * bytes;
        let kernels = Kernels::new(&host)?;
        let sync = Arc::new(Timelines::new(&host)?);
        let imports = Imports::default();
        let mut raw = Vec::new();
        let mut ring = Vec::new();
        for _ in 0..3 {
            let buffers = (0..4)
                .map(|_| Buffer::new(&host, OFFSET + bytes, false))
                .collect::<Result<Vec<_>>>()?;
            ring.push(
                buffers
                    .iter()
                    .map(|b| imports.import(&context, b))
                    .collect::<Result<Vec<_>>>()?,
            );
            raw.push(buffers);
        }
        assert!(imports.import(&context, &raw[0][0]).is_err());
        // This independently created device is ONLY a negative test. It is never
        // used as a replacement for the successfully imported host device.
        let (other_device, other_queue) =
            pollster::block_on(context.adapter.request_device(&Default::default()))?;
        let foreign = Context {
            device: other_device,
            queue: other_queue,
            adapter: context.adapter.clone(),
            c_device: std::ptr::null_mut(),
            lock: Arc::new(queue_lock::QueueLock::default()),
        };
        assert!(imports.import(&foreign, &raw[0][0]).is_err());
        drop(foreign);
        let references = (0..3)
            .map(|_| Buffer::new(&host, OFFSET + bytes, false))
            .collect::<Result<Vec<_>>>()?;
        let reference_views = references
            .iter()
            .map(|b| imports.import(&context, b))
            .collect::<Result<Vec<_>>>()?;
        let checks = Buffer::new(&host, report_bytes, true)?;
        let ptr =
            host.device
                .map_memory(checks.memory, 0, report_bytes, vk::MemoryMapFlags::empty())?;
        std::ptr::write_bytes(ptr.cast::<u8>(), 0xff, report_bytes as usize);
        host.device.unmap_memory(checks.memory);
        let model = Model::from_tza(&std::fs::read(&options.weights)?)?;
        if model.input_count() != 3 {
            return Err("requires color/albedo/normal model".into());
        }
        let mut config = DenoiserOptions::default();
        config.max_tile_size = options.tile;
        // Safety for --unchecked: the existing generated denoise kernels must
        // remain in bounds and initialize workgroup data, as in the C API.
        let mut denoiser = if options.unchecked {
            Denoiser::new_unchecked(&context.device, &context.adapter, &model, config)?
        } else {
            Denoiser::new(&context.device, &context.adapter, &model, config)?
        };
        let mut prepared = denoiser.prepare([options.size; 2], bindings(&ring[0], None))?;
        if options.trace {
            eprintln!("prepared: {:?}", prepared.info());
        }
        assert_eq!(prepared.info().runtime_checks, !options.unchecked);
        let precision = format!("{:?}", prepared.info().precision);
        let kernel_paths = format!("{:?}", prepared.info().kernels);
        for pattern in 0..3 {
            let batch = Batch::new(
                &kernels,
                false,
                [
                    (&raw[0][0], OFFSET, bytes),
                    (&raw[0][1], OFFSET, bytes),
                    (&raw[0][2], OFFSET, bytes),
                ],
                [options.size, pattern as u32, 0, 0],
            )?;
            submit(
                &host,
                host.queues[0],
                Some(&batch),
                &[],
                &[(sync.sems[0], pattern as u64 + 1)],
            )?;
            sync.wait(0, pattern as u64 + 1, TIMEOUT)?;
            prepared.rebind(bindings(&ring[0], Some(&reference_views[pattern])))?;
            let mut encoder = context.device.create_command_encoder(&Default::default());
            prepared.encode(
                &mut encoder,
                DenoiseParams {
                    input_scale: SCALES[pattern],
                },
            )?;
            context.wait(library_submit(
                &context,
                &host,
                Some(encoder.finish()),
                &[(sync.sems[0], pattern as u64 + 1)],
                (sync.sems[1], pattern as u64 + 1),
            )?)?;
        }
        let base = 3u64;
        // CPU-only rejection happens before accepting any synchronization payload.
        // Completion of wgpu's fence need not make an independent external
        // semaphore counter immediately observable to the host.
        sync.wait(1, 3, TIMEOUT)?;
        let before = sync.value(1)?;
        let mut encoder = context.device.create_command_encoder(&Default::default());
        assert!(
            prepared
                .encode(
                    &mut encoder,
                    DenoiseParams {
                        input_scale: f32::NAN
                    }
                )
                .is_err()
        );
        drop(encoder);
        let mut invalid = bindings(&ring[0], None);
        invalid.color.offset = 1;
        assert!(prepared.rebind(invalid).is_err());
        let mut in_place = bindings(&ring[0], None);
        in_place.output = in_place.color;
        assert!(prepared.rebind(in_place).is_err());
        assert_eq!(sync.value(1)?, before);
        // A controlled encode failure after accepting a valid sync description is
        // represented by an empty batch with the SAME wait and signal. No pixels
        // from this failed execution are consumed.
        let fallback = base + 1;
        let encoded: Result<wgpu::CommandBuffer> = (|| {
            let _encoder = context.device.create_command_encoder(&Default::default());
            Err("controlled encoding failure after sync acceptance".into())
        })();
        assert!(encoded.is_err());
        submit(
            &host,
            host.queues[0],
            None,
            &[(sync.sems[6], 1)],
            &[(sync.sems[0], fallback)],
        )?;
        let release = sync.clone();
        let failed_release = std::thread::spawn(move || -> std::result::Result<u64, String> {
            std::thread::sleep(Duration::from_millis(100));
            let before = release.value(1).map_err(|e| e.to_string())?;
            release.signal(6).map_err(|e| e.to_string())?;
            Ok(before)
        });
        context.wait(library_submit(
            &context,
            &host,
            encoded.ok(),
            &[(sync.sems[0], fallback)],
            (sync.sems[1], fallback),
        )?)?;
        sync.wait(1, fallback, TIMEOUT)?;
        assert_eq!(sync.value(1)?, fallback);
        let fallback_early = failed_release
            .join()
            .map_err(|_| "failure gate thread panicked")?
            .map_err(|e| format!("failure gate: {e}"))?;
        assert_eq!(fallback_early, base, "empty fallback ignored its wait");
        let mut c_api = if options.c_api {
            let mut handles = Vec::new();
            let mut groups = Vec::new();
            let mut error = [0i8; 4096];
            for group in &raw {
                let mut images = Vec::new();
                for buffer in group {
                    let create = vk::BufferCreateInfo::default()
                        .size(buffer.size)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
                    let mut handle = std::ptr::null_mut();
                    if oidnw_gpu_buffer_import_vulkan(
                        context.c_device,
                        buffer.raw,
                        &create,
                        &mut handle,
                        error.as_mut_ptr(),
                        error.len(),
                    ) != 0
                    {
                        return Err(CStr::from_ptr(error.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                            .into());
                    }
                    handles.push(handle);
                    images.push(Image {
                        buffer: handle,
                        offset: OFFSET,
                    });
                }
                groups.push(Images {
                    color: images[0],
                    albedo: images[1],
                    normal: images[2],
                    output: images[3],
                });
            }
            let mut kind =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let create = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            let mut ready = std::ptr::null_mut();
            let mut done = std::ptr::null_mut();
            for (native, handle) in [(sync.sems[0], &mut ready), (sync.sems[1], &mut done)] {
                if oidnw_gpu_sync_import_vulkan(
                    context.c_device,
                    native,
                    &create,
                    handle,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
                {
                    return Err(CStr::from_ptr(error.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                        .into());
                }
            }
            Some(gpu_client::GpuClient::new(
                context.c_device,
                groups,
                handles,
                ready,
                done,
                &std::fs::read(&options.weights)?,
                options.size,
                options.tile,
                options.unchecked,
            )?)
        } else {
            None
        };
        let signal_base = fallback;
        let release = sync.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || -> std::result::Result<(bool, u64), String> {
            let submitted = receive.recv_timeout(Duration::from_secs(10)).is_ok();
            std::thread::sleep(Duration::from_millis(150));
            let early = release.value(1).map_err(|e| e.to_string())?;
            release.signal(3).map_err(|e| e.to_string())?;
            // A separate delayed consumer gate tests output reuse dependencies.
            std::thread::sleep(Duration::from_millis(100));
            release.signal(4).map_err(|e| e.to_string())?;
            Ok((submitted, early))
        });
        assert!(sync.wait(3, 1, 0).is_err()); // deterministic timeout, no GPU fault
        let mut pending: VecDeque<(u64, Batch, Batch)> = VecDeque::new();
        let mut high_water = 0;
        let mut allocator_samples = vec![allocator_sample(&context.device, 0)];
        let mut last = None;
        for frame in 0..options.frames {
            // HAL 30 chains submissions with binary relay semaphores. Vulkan
            // forbids a binary wait whose transitive timeline signal has not been
            // submitted (03238). Test delayed signals on the first execution, then
            // resolve its CPU gates before chaining another HAL submit. Normal
            // frames still use GPU producer/consumer dependencies without CPU waits.
            if frame == 1 {
                sync.wait(4, 1, TIMEOUT)?;
            }
            {
                let _guard = host.lock.lock().unwrap();
                context.device.poll(wgpu::PollType::Poll)?;
            }
            let complete = sync.value(2)?;
            while pending.front().is_some_and(|b| b.0 <= complete) {
                pending.pop_front();
            }
            // Bounded host command/descriptor retirement, with targeted backpressure
            // at a depth of 64. No per-frame wait and no queue/device-idle call.
            if pending.len() == 64 {
                sync.wait(2, pending.front().unwrap().0, TIMEOUT)?;
                let complete = sync.value(2)?;
                while pending.front().is_some_and(|b| b.0 <= complete) {
                    pending.pop_front();
                }
            }
            let slot = frame % 3;
            let pattern = (frame + frame / 3) % 3;
            let sequence = frame as u64 + 1;
            let images = &raw[slot];
            let producer = Batch::new(
                &kernels,
                false,
                [
                    (&images[0], OFFSET, bytes),
                    (&images[1], OFFSET, bytes),
                    (&images[2], OFFSET, bytes),
                ],
                [options.size, pattern as u32, 0, 0],
            )?;
            let mut waits = vec![(sync.sems[3], 1)];
            if frame >= 3 {
                waits.push((sync.sems[2], sequence - 3));
            }
            if frame + 1 == options.frames {
                waits.push((sync.sems[5], 1));
            }
            submit(
                &host,
                host.queues[0],
                Some(&producer),
                &waits,
                &[(sync.sems[0], fallback + sequence)],
            )?;
            if let Some(api) = c_api.as_mut() {
                api.submit(slot, fallback + sequence, SCALES[pattern])?;
            } else {
                prepared.rebind(bindings(&ring[slot], None))?;
                let mut encoder = context.device.create_command_encoder(&Default::default());
                prepared.encode(
                    &mut encoder,
                    DenoiseParams {
                        input_scale: SCALES[pattern],
                    },
                )?;
                last = Some(library_submit(
                    &context,
                    &host,
                    Some(encoder.finish()),
                    &[(sync.sems[0], fallback + sequence)],
                    (sync.sems[1], signal_base + sequence),
                )?);
            }
            let consumer = Batch::new(
                &kernels,
                true,
                [
                    (&images[3], OFFSET, bytes),
                    (&references[pattern], OFFSET, bytes),
                    (&checks, 0, report_bytes),
                ],
                [options.size, pattern as u32, frame as u32, 0],
            )?;
            submit(
                &host,
                host.queues[2],
                Some(&consumer),
                &[(sync.sems[1], signal_base + sequence), (sync.sems[4], 1)],
                &[(sync.sems[2], sequence)],
            )?;
            pending.push_back((sequence, producer, consumer));
            high_water = high_water.max(pending.len());
            if (frame + 1) % 250 == 0 {
                allocator_samples.push(allocator_sample(&context.device, frame + 1));
            }
            if frame == 0 {
                send.send(())?;
            }
        }
        let (submitted, early) = thread
            .join()
            .map_err(|_| "gate thread panicked")?
            .map_err(|e| format!("gate: {e}"))?;
        if !submitted || early != signal_base {
            return Err(
                format!("producer gate failed: submitted={submitted}, early={early}").into(),
            );
        }
        // Release every imported application handle and the prepared workspace while
        // commands remain in flight. HAL callbacks retain native allocations.
        if let Some(api) = c_api.as_mut() {
            api.release_handles();
        }
        drop(prepared);
        drop(denoiser);
        drop(ring);
        drop(reference_views);
        let callbacks_before_release = imports.drops.load(Ordering::Relaxed);
        assert!(
            options.c_api || callbacks_before_release < 15,
            "in-flight imported buffers released early"
        );
        sync.signal(5)?;
        let c_queue_full = c_api.as_ref().map_or(0, |api| api.queue_full);
        if let Some(api) = c_api.take() {
            api.finish()?;
        } else {
            context.wait(last.unwrap())?;
        }
        sync.wait(2, options.frames as u64, TIMEOUT)?;
        pending.clear();
        {
            let _guard = host.lock.lock().unwrap();
            context.device.poll(wgpu::PollType::Poll)?;
        }
        let ptr =
            host.device
                .map_memory(checks.memory, 0, report_bytes, vk::MemoryMapFlags::empty())?;
        let flags = std::slice::from_raw_parts(ptr.cast::<u32>(), (report_bytes / 4) as usize);
        let failure = flags
            .iter()
            .enumerate()
            .find(|(_, f)| **f != 0)
            .map(|(i, f)| (i, *f));
        host.device.unmap_memory(checks.memory);
        if let Some(f) = failure {
            return Err(format!("GPU consumer mismatch (word, flags): {f:?}").into());
        }
        let name = context.adapter.get_info().name;
        // Include wgpu/HAL teardown under the same host queue lock. No host worker
        // may touch a shared queue outside this lock, including during destruction.
        {
            let _guard = host.lock.lock().unwrap();
            drop(context);
        }
        assert!(imports.live.lock().unwrap().is_empty());
        // Check validation after native children, device and instance teardown,
        // so a leaked object or an accidental host-resource destruction cannot
        // appear only after the success report.
        drop(checks);
        drop(raw);
        drop(references);
        drop(kernels);
        drop(sync);
        drop(host);
        if !errors.lock().unwrap().is_empty() || VALIDATION_ERRORS.load(Ordering::Relaxed) != 0 {
            return Err(format!("GPU validation: {:?}", errors.lock().unwrap()).into());
        }
        println!(
            "{}",
            serde_json::json!({"backend":"Vulkan", "route":"host-device-import", "device_created_by":"host",
        "adapter":name,"frames":options.frames,"size":options.size,"tile":options.tile,"precision":precision,"kernels":kernel_paths,
        "formal_c_api":options.c_api,"queue_full":c_queue_full,"runtime_checks":!options.unchecked,
        "queue":if options.shared {"shared"} else {"dedicated"},"requirements":requirements,"validation_errors":VALIDATION_ERRORS.load(Ordering::Relaxed),
        "known_naga_workgroup_layout_diagnostics":NAGA_LAYOUT_ERRORS.load(Ordering::Relaxed),
        "output_byte_equal":true,"completed_before_producer_release":early - signal_base,"cpu_image_uploads":0,"cpu_image_readbacks":0,
        "pending_batches_high_water":high_water,"pending_batches_after_wait":pending.len(),"import_drop_callbacks":imports.drops.load(Ordering::Relaxed),
        "allocator_samples":allocator_samples,
        "import_callbacks_before_final_gate_release":callbacks_before_release,
        "invalid_parameter_rejected":true,"in_place_rejected":true,"duplicate_import_rejected":true,"wrong_device_rejected":true,"empty_failure_batch_signalled":true,"timeout_checked":true})
        );
        Ok(())
    }
}
