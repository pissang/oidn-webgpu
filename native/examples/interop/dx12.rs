#[allow(dead_code)]
#[path = "common.rs"]
mod common;
use common::*;
#[path = "gpu_client.rs"]
mod gpu_client;
use oidn_webgpu::gpu_capi::*;
use oidn_webgpu::{DenoiseParams, Denoiser, DenoiserOptions, Model, wgpu};
use std::{
    collections::{HashSet, VecDeque},
    ffi::{CString, c_void},
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        Graphics::{Direct3D::Fxc::*, Direct3D::*, Direct3D12::*, Dxgi::Common::*, Dxgi::*},
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
    core::{Interface, PCSTR, PCWSTR},
};
type Api = wgpu::hal::api::Dx12;

struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}
fn same_luid(a: windows::Win32::Foundation::LUID, b: windows::Win32::Foundation::LUID) -> bool {
    a.HighPart == b.HighPart && a.LowPart == b.LowPart
}
fn validate_luid(
    a: windows::Win32::Foundation::LUID,
    b: windows::Win32::Foundation::LUID,
) -> Result<()> {
    if !same_luid(a, b) {
        return Err("adapter LUID mismatch".into());
    }
    Ok(())
}
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    raw: ID3D12Device,
    c_device: *mut GpuDevice,
}
impl Context {
    fn wait(&self, index: wgpu::SubmissionIndex) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(Duration::from_secs(60)),
        })?;
        Ok(())
    }
}
struct Host {
    device: ID3D12Device,
    producer: ID3D12CommandQueue,
    consumer: ID3D12CommandQueue,
    _layers: Option<libloading::Library>,
}

unsafe fn setup(options: &Options) -> Result<(Host, Context, bool)> {
    unsafe {
        // Agility's Independent Devices API permits two genuinely separate devices
        // and provides a redistributable debug layer without installing OS features.
        let sdk = wgpu::Dx12BackendOptions::default().with_env().agility_sdk.ok_or("set WGPU_DX12_AGILITY_SDK_PATH and WGPU_DX12_AGILITY_SDK_VERSION to a local Agility SDK (debug layers required)")?;
        let sdk_dir = std::path::PathBuf::from(&sdk.sdk_path);
        let sdk_dir = if sdk_dir.is_absolute() {
            sdk_dir
        } else {
            std::env::current_exe()?.parent().unwrap().join(sdk_dir)
        };
        let layers = if options.debug {
            Some(
                libloading::Library::new(sdk_dir.join("d3d12SDKLayers.dll"))
                    .map_err(|e| format!("load Agility SDK layers: {e}"))?,
            )
        } else {
            None
        };
        let mut config = None;
        D3D12GetInterface::<ID3D12SDKConfiguration1>(&CLSID_D3D12SDKConfiguration, &mut config)?;
        let path = CString::new(sdk.sdk_path.as_bytes())?;
        let factory: ID3D12DeviceFactory = config
            .unwrap()
            .CreateDeviceFactory(sdk.sdk_version, PCSTR(path.as_ptr().cast()))
            .map_err(|e| format!("Agility CreateDeviceFactory: {e}"))?;
        let factory_desc = factory.cast::<ID3D12DeviceConfiguration>()?.GetDesc();
        if options.trace {
            eprintln!("host factory SDK={}", factory_desc.SDKVersion);
        }
        let debug: Option<ID3D12Debug> = if options.debug {
            Some(factory.GetConfigurationInterface(&CLSID_D3D12Debug)
        .map_err(|e|format!("Agility debug interface: {e}; install Windows Graphics Tools, or explicitly use --no-debug for an unvalidated diagnostic run"))?)
        } else {
            None
        };
        if let Some(debug) = &debug {
            debug.EnableDebugLayer();
        }
        let gpu_validation = std::env::var("WGPU_GPU_BASED_VALIDATION").is_ok_and(|s| s == "1");
        if gpu_validation {
            debug
                .as_ref()
                .ok_or("GPU validation requires debug layer")?
                .cast::<ID3D12Debug1>()?
                .SetEnableGPUBasedValidation(true);
        }
        // D3D12 SDK layers come from Agility. DXGI's separate OS debug component is
        // optional; do not require its global installation to test D3D12 validation.
        let dxgi: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let mut selected = None;
        for index in 0..32 {
            match dxgi.EnumAdapters1(index) {
                Ok(adapter) => {
                    let desc = adapter.GetDesc1()?;
                    let name = String::from_utf16_lossy(&desc.Description);
                    if name.contains(&options.adapter) {
                        selected = Some((adapter, desc));
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let (adapter, desc) = selected.ok_or("host DXGI adapter not found")?;
        let mut host_device = None;
        factory.CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut host_device)?;
        let host_device: ID3D12Device = host_device.ok_or("host device creation returned null")?;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::DX12;
        descriptor.flags = if options.debug {
            wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION
        } else {
            wgpu::InstanceFlags::empty()
        };
        if gpu_validation {
            descriptor.flags |= wgpu::InstanceFlags::GPU_BASED_VALIDATION;
        }
        descriptor.backend_options.dx12.agility_sdk = Some(sdk);
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12))
            .into_iter()
            .find(|a| {
                a.as_hal::<Api>().is_some_and(|hal| {
                    same_luid(
                        hal.raw_adapter().GetDesc().unwrap().AdapterLuid,
                        desc.AdapterLuid,
                    )
                })
            })
            .ok_or("no wgpu adapter matches host LUID")?;
        let mut features = if options.recommended {
            Denoiser::recommended_features(&adapter)
        } else {
            Denoiser::required_features()
        };
        if options.fp32 {
            features.remove(wgpu::Features::SHADER_F16);
        }
        // Current DX12 implicit GEMM only; no experimental matrix feature.
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: Denoiser::required_limits(),
                ..Default::default()
            }))?;
        let raw = device.as_hal::<Api>().unwrap().raw_device().clone();
        for (name, raw_device) in [("host", &host_device), ("library", &raw)] {
            let flags = raw_device
                .cast::<ID3D12DeviceConfiguration>()?
                .GetDesc()
                .Flags;
            if options.debug && !flags.contains(D3D12_DEVICE_FLAG_DEBUG_LAYER_ENABLED) {
                return Err(
                    format!("{name} device did not enable the requested debug layer").into(),
                );
            }
            if gpu_validation && !flags.contains(D3D12_DEVICE_FLAG_GPU_BASED_VALIDATION_ENABLED) {
                return Err(
                    format!("{name} device did not enable the requested GPU validation").into(),
                );
            }
        }
        validate_luid(host_device.GetAdapterLuid(), raw.GetAdapterLuid())?;
        if host_device.as_raw() == raw.as_raw() {
            return Err("external-memory test requires distinct host/library devices".into());
        }
        let q = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        let producer = host_device.CreateCommandQueue(&q)?;
        let consumer = host_device.CreateCommandQueue(&q)?;
        let (adapter, device, queue, c_device) = if options.c_api {
            #[cfg(not(feature = "interop-validation"))]
            {
                return Err("--c-api requires --features interop-validation".into());
            }
            #[cfg(feature = "interop-validation")]
            {
                let mut c_device = std::ptr::null_mut();
                let desc = Dx12DeviceDesc {
                    flags: u32::from(options.fp32)
                        | if !options.recommended { 3 } else { 0 }
                        | if options.debug { 4 } else { 0 }
                        | if gpu_validation { 8 } else { 0 },
                    host_device: host_device.as_raw(),
                    agility_sdk_version: factory_desc.SDKVersion,
                    agility_sdk_path: path.as_ptr(),
                    max_pending_submissions: 8,
                };
                let mut error = [0i8; 4096];
                if oidnw_gpu_device_create_dx12(
                    &desc,
                    &mut c_device,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
                {
                    return Err(std::ffi::CStr::from_ptr(error.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                        .into());
                }
                let (a, d, q) = reference_context(c_device)
                    .map_err(|e| format!("C API reference device: {e}"))?;
                (a, d, q, c_device)
            }
        } else {
            (adapter, device, queue, std::ptr::null_mut())
        };
        let raw = device.as_hal::<Api>().unwrap().raw_device().clone();
        Ok((
            Host {
                device: host_device,
                producer,
                consumer,
                _layers: layers,
            },
            Context {
                device,
                queue,
                adapter,
                raw,
                c_device,
            },
            gpu_validation,
        ))
    }
}
unsafe fn buffer(
    device: &ID3D12Device,
    bytes: u64,
    shared: bool,
    readback: bool,
) -> Result<ID3D12Resource> {
    unsafe {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback {
                D3D12_HEAP_TYPE_READBACK
            } else {
                D3D12_HEAP_TYPE_DEFAULT
            },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if readback {
                D3D12_RESOURCE_FLAG_NONE
            } else {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            },
            ..Default::default()
        };
        let mut result = None;
        device.CreateCommittedResource(
            &heap,
            if shared {
                D3D12_HEAP_FLAG_SHARED
            } else {
                D3D12_HEAP_FLAG_NONE
            },
            &desc,
            if readback {
                D3D12_RESOURCE_STATE_COPY_DEST
            } else {
                D3D12_RESOURCE_STATE_COMMON
            },
            None,
            &mut result,
        )?;
        Ok(result.unwrap())
    }
}
#[derive(Default)]
struct Imports {
    ids: HashSet<usize>,
}
impl Imports {
    unsafe fn import(
        &mut self,
        context: &Context,
        host: &Host,
        resource: &ID3D12Resource,
    ) -> Result<wgpu::Buffer> {
        unsafe {
            validate_luid(host.device.GetAdapterLuid(), context.raw.GetAdapterLuid())?;
            let mut origin = None;
            resource.GetDevice::<ID3D12Device>(&mut origin)?;
            if origin.unwrap().as_raw() != host.device.as_raw() {
                return Err("resource is not owned by the exporting host device".into());
            }
            let desc = resource.GetDesc();
            if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                || !desc
                    .Flags
                    .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
            {
                return Err("shared resource must be a UAV-capable buffer".into());
            }
            let id = resource.as_raw() as usize;
            if self.ids.contains(&id) {
                return Err("duplicate native resource import".into());
            }
            // Exporter owns each NT handle and closes it after OpenSharedHandle.
            // Opening obtains a COM reference; it does not transfer handle ownership.
            let handle = Handle(host.device.CreateSharedHandle(
                resource,
                None,
                0x10000000,
                PCWSTR::null(),
            )?);
            let mut opened = None;
            context
                .raw
                .OpenSharedHandle::<ID3D12Resource>(handle.0, &mut opened)?;
            drop(handle);
            let opened = opened.unwrap();
            let hal = wgpu::hal::dx12::Device::buffer_from_raw(opened, desc.Width);
            let wrapped = context.device.create_buffer_from_hal::<Api>(
                hal,
                &wgpu::BufferDescriptor {
                    label: Some("opened shared host DX12 buffer"),
                    size: desc.Width,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            );
            self.ids.insert(id);
            Ok(wrapped)
        }
    }
}
struct Fences {
    host: [ID3D12Fence; 8],
    library: [ID3D12Fence; 8],
}
impl Fences {
    unsafe fn new(host: &Host, context: &Context) -> Result<Self> {
        unsafe {
            let mut exported = Vec::new();
            let mut imported = Vec::new();
            for _ in 0..8 {
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_SHARED)?;
                let handle = Handle(host.device.CreateSharedHandle(
                    &fence,
                    None,
                    0x10000000,
                    PCWSTR::null(),
                )?);
                let mut opened = None;
                context
                    .raw
                    .OpenSharedHandle::<ID3D12Fence>(handle.0, &mut opened)?;
                exported.push(fence);
                imported.push(opened.unwrap());
            }
            Ok(Self {
                host: exported.try_into().unwrap(),
                library: imported.try_into().unwrap(),
            })
        }
    }
    fn value(&self, i: usize) -> Result<u64> {
        let v = unsafe { self.host[i].GetCompletedValue() };
        if v == u64::MAX {
            return Err("host device lost; fence value is UINT64_MAX".into());
        }
        Ok(v)
    }
    fn signal(&self, i: usize) -> Result<()> {
        unsafe {
            self.host[i].Signal(1)?;
        }
        Ok(())
    }
    fn wait(&self, i: usize, value: u64, timeout: u32) -> Result<()> {
        unsafe {
            if self.value(i)? >= value {
                return Ok(());
            }
            let event = Handle(CreateEventW(None, false, false, PCWSTR::null())?);
            self.host[i].SetEventOnCompletion(value, event.0)?;
            if WaitForSingleObject(event.0, timeout).0 != 0 {
                return Err("fence wait timed out".into());
            }
            Ok(())
        }
    }
}

const HLSL: &str = include_str!("host.hlsl");
struct Kernels {
    root: ID3D12RootSignature,
    pipelines: [ID3D12PipelineState; 3],
}
impl Kernels {
    unsafe fn new(host: &Host) -> Result<Self> {
        unsafe {
            let mut params = Vec::new();
            for index in 0..3 {
                params.push(D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: index,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                });
            }
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                ..Default::default()
            };
            let mut blob = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
            let blob = blob.unwrap();
            let root: ID3D12RootSignature = host.device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()),
            )?;
            let mut pipelines = Vec::new();
            for entry in [c"produce", c"consume", c"initialize"] {
                let mut code = None;
                let mut errors = None;
                if let Err(e) = D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    PCSTR::null(),
                    None,
                    None,
                    PCSTR(entry.as_ptr().cast()),
                    windows::core::s!("cs_5_1"),
                    D3DCOMPILE_ENABLE_STRICTNESS,
                    0,
                    &mut code,
                    Some(&mut errors),
                ) {
                    if let Some(errors) = errors {
                        eprintln!(
                            "{}",
                            String::from_utf8_lossy(std::slice::from_raw_parts(
                                errors.GetBufferPointer().cast(),
                                errors.GetBufferSize()
                            ))
                        );
                    }
                    return Err(e.into());
                }
                let code = code.unwrap();
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: ManuallyDrop::new(Some(root.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                };
                let pipeline = host.device.CreateComputePipelineState(&desc);
                // windows-rs generated native descriptors use ManuallyDrop for COM fields.
                let mut desc = desc;
                ManuallyDrop::drop(&mut desc.pRootSignature);
                pipelines.push(pipeline?);
            }
            Ok(Self {
                root,
                pipelines: pipelines.try_into().unwrap(),
            })
        }
    }
}
// Every buffer command list implicitly promotes COMMON -> UAV and decays to
// COMMON at ExecuteCommandLists completion. This applies to buffers on DIRECT
// queues too. Input/output roles remain fixed across frames, including in wgpu's
// state tracker; fence ordering prevents cross-device overlap. Native UAV
// barriers order writes before command-list completion and before validation copy.
struct Batch {
    _allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}
unsafe fn uav(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    unsafe {
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                }),
            },
        };
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.UAV).pResource);
    }
}
impl Batch {
    unsafe fn initialize(
        host: &Host,
        k: &Kernels,
        images: &[ID3D12Resource],
        checks: &ID3D12Resource,
        count: u32,
    ) -> Result<Self> {
        unsafe {
            let allocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                &k.pipelines[2],
            )?;
            list.SetComputeRootSignature(&k.root);
            list.SetComputeRootUnorderedAccessView(0, images[0].GetGPUVirtualAddress() + OFFSET);
            list.SetComputeRootUnorderedAccessView(1, images[1].GetGPUVirtualAddress() + OFFSET);
            list.SetComputeRootUnorderedAccessView(2, checks.GetGPUVirtualAddress());
            let params = [count, 0, 0, 0];
            list.SetComputeRoot32BitConstants(3, 4, params.as_ptr().cast(), 0);
            let groups = count.div_ceil(64);
            list.Dispatch(groups.min(65535), groups.div_ceil(65535), 1);
            uav(&list, checks);
            list.Close()?;
            Ok(Self {
                _allocator: allocator,
                list,
            })
        }
    }
    unsafe fn dispatch(
        host: &Host,
        k: &Kernels,
        consume: bool,
        buffers: [&ID3D12Resource; 3],
        params: [u32; 4],
    ) -> Result<Self> {
        unsafe {
            let allocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                &k.pipelines[consume as usize],
            )?;
            list.SetComputeRootSignature(&k.root);
            for (index, buffer) in buffers.iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(
                    index as u32,
                    buffer.GetGPUVirtualAddress() + if index == 2 && consume { 0 } else { OFFSET },
                );
            }
            list.SetComputeRoot32BitConstants(3, 4, params.as_ptr().cast(), 0);
            list.Dispatch((params[0] * params[0]).div_ceil(64), 1, 1);
            if consume {
                uav(&list, buffers[2]);
            } else {
                for buffer in buffers {
                    uav(&list, buffer);
                }
            }
            list.Close()?;
            Ok(Self {
                _allocator: allocator,
                list,
            })
        }
    }
    unsafe fn copy(
        host: &Host,
        source: &ID3D12Resource,
        dest: &ID3D12Resource,
        bytes: u64,
    ) -> Result<Self> {
        unsafe {
            let allocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            // The preceding consumer list has already decayed its buffer to COMMON.
            // A distinct ExecuteCommandLists call permits implicit COPY_SOURCE promotion.
            list.CopyBufferRegion(dest, 0, source, 0, bytes);
            list.Close()?;
            Ok(Self {
                _allocator: allocator,
                list,
            })
        }
    }
}
unsafe fn submit(
    queue: &ID3D12CommandQueue,
    batch: Option<&Batch>,
    waits: &[(&ID3D12Fence, u64)],
    signals: &[(&ID3D12Fence, u64)],
) -> Result<()> {
    unsafe {
        for &(f, v) in waits {
            queue.Wait(f, v)?;
        }
        if let Some(batch) = batch {
            queue.ExecuteCommandLists(&[Some(batch.list.cast()?)]);
        }
        for &(f, v) in signals {
            queue.Signal(f, v)?;
        }
        Ok(())
    }
}
unsafe fn library_submit(
    context: &Context,
    command: Option<wgpu::CommandBuffer>,
    sync: &Fences,
    ready: u64,
    done: u64,
) -> wgpu::SubmissionIndex {
    unsafe {
        let hal = context.queue.as_hal::<Api>().unwrap();
        hal.add_wait_fence(sync.library[0].clone(), ready);
        hal.add_signal_fence(sync.library[1].clone(), done);
        drop(hal);
        context.queue.submit(command)
    }
}
unsafe fn validation(device: &ID3D12Device) -> Result<usize> {
    unsafe {
        let info: ID3D12InfoQueue = device.cast()?;
        let mut failures = 0;
        for index in 0..info.GetNumStoredMessagesAllowedByRetrievalFilter() {
            let mut bytes = 0;
            info.GetMessage(index, None, &mut bytes)?;
            let mut storage = vec![0u64; bytes.div_ceil(8)];
            let ptr = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
            info.GetMessage(index, Some(ptr), &mut bytes)?;
            let message = &*ptr;
            if message.Severity == D3D12_MESSAGE_SEVERITY_ERROR
                || message.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
            {
                failures += 1;
                eprintln!(
                    "DX12 validation: {}",
                    std::ffi::CStr::from_ptr(message.pDescription.cast()).to_string_lossy()
                );
            }
        }
        Ok(failures)
    }
}

pub fn run() -> Result<()> {
    unsafe {
        let options = Options::parse()?;
        if options.shared {
            return Err(
                "DX12 prototype uses library-owned queue; --shared-queue is Vulkan-only".into(),
            );
        }
        let (host, context, gpu_validation) = setup(&options)?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let capture = errors.clone();
        context.device.on_uncaptured_error(Arc::new(move |e| {
            eprintln!(
                "unrecoverable wgpu error; shared-resource submission state is uncertain: {e}"
            );
            capture.lock().unwrap().push(e.to_string());
            // Test-process policy only. A reusable library needs an explicit
            // poisoned/uncertain submission state, rather than process exit.
            std::process::exit(2);
        }));
        let capture = errors.clone();
        context
            .device
            .set_device_lost_callback(move |reason, message| {
                if reason != wgpu::DeviceLostReason::Destroyed {
                    eprintln!(
                        "library device lost; host device state is unknown: {reason:?} {message}"
                    );
                    capture
                        .lock()
                        .unwrap()
                        .push(format!("library device lost: {reason:?} {message}"));
                    std::process::exit(2);
                }
            });
        let bytes = options.size as u64 * options.size as u64 * 16;
        let report_bytes = options.frames as u64 * bytes;
        let kernels = Kernels::new(&host)?;
        let sync = Arc::new(Fences::new(&host, &context)?);
        let mut imports = Imports::default();
        let mut raw = Vec::new();
        let mut ring = Vec::new();
        for _ in 0..3 {
            let buffers = (0..4)
                .map(|_| buffer(&host.device, OFFSET + bytes, true, false))
                .collect::<Result<Vec<_>>>()?;
            ring.push(
                buffers
                    .iter()
                    .map(|b| imports.import(&context, &host, b))
                    .collect::<Result<Vec<_>>>()?,
            );
            raw.push(buffers);
        }
        assert!(imports.import(&context, &host, &raw[0][0]).is_err());
        let foreign = buffer(&context.raw, OFFSET + bytes, true, false)?;
        assert!(imports.import(&context, &host, &foreign).is_err());
        drop(foreign);
        let bad = buffer(&host.device, bytes, false, true)?;
        assert!(imports.import(&context, &host, &bad).is_err());
        let mut wrong = host.device.GetAdapterLuid();
        wrong.LowPart ^= 1;
        assert!(validate_luid(wrong, context.raw.GetAdapterLuid()).is_err());
        let references = (0..3)
            .map(|_| buffer(&host.device, OFFSET + bytes, true, false))
            .collect::<Result<Vec<_>>>()?;
        let reference_views = references
            .iter()
            .map(|b| imports.import(&context, &host, b))
            .collect::<Result<Vec<_>>>()?;
        let checks = buffer(&host.device, report_bytes, false, false)?;
        let readback = buffer(&host.device, report_bytes, false, true)?;
        let initialize = Batch::initialize(
            &host,
            &kernels,
            &raw[0],
            &checks,
            (report_bytes / 16) as u32,
        )?;
        submit(
            &host.consumer,
            Some(&initialize),
            &[],
            &[(&sync.host[7], 1)],
        )?;
        sync.wait(7, 1, 60_000)?;
        drop(initialize);
        let model = Model::from_tza(&std::fs::read(&options.weights)?)?;
        if model.input_count() != 3 {
            return Err("requires three-input model".into());
        }
        let mut config = DenoiserOptions::default();
        config.max_tile_size = options.tile;
        // --unchecked uses the same generated kernels as the C filter default.
        // They must remain in bounds and initialize workgroup data; references
        // and asynchronous runs use the identical check setting.
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
        let paths = format!("{:?}", prepared.info().kernels);
        for pattern in 0..3 {
            let producer = Batch::dispatch(
                &host,
                &kernels,
                false,
                [&raw[0][0], &raw[0][1], &raw[0][2]],
                [options.size, pattern as u32, 0, 0],
            )?;
            submit(
                &host.producer,
                Some(&producer),
                &[],
                &[(&sync.host[0], pattern as u64 + 1)],
            )?;
            sync.wait(0, pattern as u64 + 1, 60_000)?;
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
                Some(encoder.finish()),
                &sync,
                pattern as u64 + 1,
                pattern as u64 + 1,
            ))?;
            if options.trace {
                eprintln!("reference {pattern} complete");
            }
        }
        // HAL signals its internal wgpu fence before its external fences. A
        // wgpu submission wait alone does not prove the host has observed the
        // subsequently enqueued external Signal operation (notably with GBV).
        sync.wait(1, 3, 60_000)?;
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
        let encoded: Result<wgpu::CommandBuffer> = (|| {
            let _encoder = context.device.create_command_encoder(&Default::default());
            Err("controlled encoding failure after sync acceptance".into())
        })();
        assert!(encoded.is_err());
        submit(
            &host.producer,
            None,
            &[(&sync.host[6], 1)],
            &[(&sync.host[0], 4)],
        )?;
        let release = sync.clone();
        let failed_release = std::thread::spawn(move || -> std::result::Result<u64, String> {
            std::thread::sleep(Duration::from_millis(100));
            let before = release.value(1).map_err(|e| e.to_string())?;
            release.signal(6).map_err(|e| e.to_string())?;
            Ok(before)
        });
        context.wait(library_submit(&context, encoded.ok(), &sync, 4, 4))?;
        sync.wait(1, 4, 60_000)?;
        assert_eq!(sync.value(1)?, 4);
        let fallback_early = failed_release
            .join()
            .map_err(|_| "failure gate thread panicked")?
            .map_err(|e| format!("failure gate: {e}"))?;
        assert_eq!(fallback_early, 3, "empty fallback ignored its wait");
        let mut c_api = if options.c_api {
            let mut imported = Vec::new();
            let mut images = Vec::new();
            let mut error = [0i8; 4096];
            for group in &raw {
                let mut views = Vec::new();
                for resource in group {
                    let mut handle = std::ptr::null_mut();
                    if oidnw_gpu_buffer_import_dx12(
                        context.c_device,
                        resource.as_raw(),
                        &mut handle,
                        error.as_mut_ptr(),
                        error.len(),
                    ) != 0
                    {
                        return Err(std::ffi::CStr::from_ptr(error.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                            .into());
                    }
                    imported.push(handle);
                    views.push(Image {
                        buffer: handle,
                        offset: OFFSET,
                    });
                }
                images.push(Images {
                    color: views[0],
                    albedo: views[1],
                    normal: views[2],
                    output: views[3],
                });
            }
            let mut ready = std::ptr::null_mut();
            let mut done = std::ptr::null_mut();
            for (native, handle) in [(&sync.host[0], &mut ready), (&sync.host[1], &mut done)] {
                if oidnw_gpu_sync_import_dx12(
                    context.c_device,
                    native.as_raw(),
                    handle,
                    error.as_mut_ptr(),
                    error.len(),
                ) != 0
                {
                    return Err(std::ffi::CStr::from_ptr(error.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                        .into());
                }
            }
            Some(gpu_client::GpuClient::new(
                context.c_device,
                images,
                imported,
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
        let release = sync.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || -> std::result::Result<(bool, u64), String> {
            let submitted = receive.recv_timeout(Duration::from_secs(10)).is_ok();
            std::thread::sleep(Duration::from_millis(150));
            let early = release.value(1).map_err(|e| e.to_string())?;
            release.signal(3).map_err(|e| e.to_string())?;
            std::thread::sleep(Duration::from_millis(100));
            release.signal(4).map_err(|e| e.to_string())?;
            Ok((submitted, early))
        });
        assert!(sync.wait(3, 1, 0).is_err());
        let mut pending: VecDeque<(u64, Batch, Batch)> = VecDeque::new();
        let mut high_water = 0;
        let mut allocator_samples = vec![allocator_sample(&context.device, 0)];
        let mut last = None;
        let started = Instant::now();
        for frame in 0..options.frames {
            context.device.poll(wgpu::PollType::Poll)?;
            let complete = sync.value(2)?;
            while pending.front().is_some_and(|b| b.0 <= complete) {
                pending.pop_front();
            }
            if pending.len() == 64 {
                sync.wait(2, pending.front().unwrap().0, 60_000)?;
                let complete = sync.value(2)?;
                while pending.front().is_some_and(|b| b.0 <= complete) {
                    pending.pop_front();
                }
            }
            let slot = frame % 3;
            let pattern = (frame + frame / 3) % 3;
            let sequence = frame as u64 + 1;
            let producer = Batch::dispatch(
                &host,
                &kernels,
                false,
                [&raw[slot][0], &raw[slot][1], &raw[slot][2]],
                [options.size, pattern as u32, 0, 0],
            )?;
            let mut waits = vec![(&sync.host[3], 1)];
            if frame >= 3 {
                waits.push((&sync.host[2], sequence - 3));
            }
            if frame + 1 == options.frames {
                waits.push((&sync.host[5], 1));
            }
            submit(
                &host.producer,
                Some(&producer),
                &waits,
                &[(&sync.host[0], 4 + sequence)],
            )?;
            if let Some(api) = c_api.as_mut() {
                api.submit(slot, 4 + sequence, SCALES[pattern])?;
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
                    Some(encoder.finish()),
                    &sync,
                    4 + sequence,
                    4 + sequence,
                ));
            }
            let consumer = Batch::dispatch(
                &host,
                &kernels,
                true,
                [&raw[slot][3], &references[pattern], &checks],
                [options.size, pattern as u32, frame as u32, 0],
            )?;
            submit(
                &host.consumer,
                Some(&consumer),
                &[(&sync.host[1], 4 + sequence), (&sync.host[4], 1)],
                &[(&sync.host[2], sequence)],
            )?;
            pending.push_back((sequence, producer, consumer));
            high_water = high_water.max(pending.len());
            if (frame + 1) % 250 == 0 {
                allocator_samples.push(allocator_sample(&context.device, frame + 1));
            }
            if frame + 1 == options.frames.min(3) {
                send.send(())?;
            }
            if options.trace && (frame + 1) % 250 == 0 {
                eprintln!(
                    "submitted {} frames, consumed={}, pending={}",
                    frame + 1,
                    sync.value(2)?,
                    pending.len()
                );
            }
        }
        let (submitted, early) = thread
            .join()
            .map_err(|_| "gate thread panic")?
            .map_err(|e| format!("gate: {e}"))?;
        if !submitted || early != 4 {
            return Err(format!("producer gate failed: {submitted}, {early}").into());
        }
        if let Some(api) = c_api.as_mut() {
            api.release_handles();
        }
        drop(prepared);
        drop(denoiser);
        drop(ring);
        drop(reference_views);
        // Last producer is still gated here; opened COM resource references in wgpu
        // commands must outlive the application handles just released.
        assert!(sync.value(1)? < 4 + options.frames as u64);
        sync.signal(5)?;
        let c_queue_full = c_api.as_ref().map_or(0, |api| api.queue_full);
        if let Some(api) = c_api.take() {
            api.finish()?;
        } else {
            context.wait(last.unwrap())?;
        }
        sync.wait(2, options.frames as u64, 60_000)?;
        pending.clear();
        let copy = Batch::copy(&host, &checks, &readback, report_bytes)?;
        submit(
            &host.consumer,
            Some(&copy),
            &[],
            &[(&sync.host[2], options.frames as u64 + 1)],
        )?;
        sync.wait(2, options.frames as u64 + 1, 60_000)?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        readback.Map(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: report_bytes as usize,
            }),
            Some(&mut ptr),
        )?;
        let flags = std::slice::from_raw_parts(ptr.cast::<u32>(), (report_bytes / 4) as usize);
        let failure = flags
            .iter()
            .enumerate()
            .find(|(_, v)| **v != 0)
            .map(|(i, v)| (i, *v));
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        if let Some(failure) = failure {
            return Err(format!("GPU consumer mismatch (word, flags): {failure:?}").into());
        }
        drop(copy);
        drop(readback);
        drop(checks);
        drop(raw);
        drop(references);
        drop(kernels);
        drop(sync);
        drop(bad);
        context.device.poll(wgpu::PollType::Poll)?;
        let validation_errors = if options.debug {
            validation(&host.device)? + validation(&context.raw)?
        } else {
            0
        };
        if validation_errors != 0 || !errors.lock().unwrap().is_empty() {
            return Err(format!(
                "GPU errors: {validation_errors}, {:?}",
                errors.lock().unwrap()
            )
            .into());
        }
        let luid = host.device.GetAdapterLuid();
        println!(
            "{}",
            serde_json::json!({"backend":"DX12","route":"external-memory","device_created_by":"library",
        "adapter":context.adapter.get_info().name,"adapter_luid":[luid.HighPart as i64,luid.LowPart as i64],"distinct_devices":true,
        "frames":options.frames,"size":options.size,"tile":options.tile,"precision":precision,"kernels":paths,
        "formal_c_api":options.c_api,"queue_full":c_queue_full,"runtime_checks":!options.unchecked,
        "output_byte_equal":true,"completed_before_producer_release":early-4,"cpu_image_uploads":0,"cpu_image_readbacks":0,
        "pending_batches_high_water":high_water,"pending_batches_after_wait":pending.len(),"debug_layer":options.debug,"gpu_validation":gpu_validation,
        "allocator_samples":allocator_samples,
        "validation_errors":if options.debug {Some(validation_errors)} else {None},"invalid_parameter_rejected":true,"duplicate_import_rejected":true,"wrong_luid_rejected":true,
        "wrong_exporter_device_rejected":true,"in_place_rejected":true,
        "non_uav_rejected":true,"empty_failure_batch_signalled":true,"timeout_checked":true,"elapsed_including_delays_ms":started.elapsed().as_secs_f64()*1000.0})
        );
        Ok(())
    }
}
