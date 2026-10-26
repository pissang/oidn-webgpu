use super::*;
use windows::{
    Win32::Graphics::{Direct3D::*, Direct3D12::*, Dxgi::Common::*, Dxgi::*},
    core::{Interface, PCSTR},
};

#[test]
#[ignore = "requires Windows GPU and matching Agility SDK; shaders explicitly unchecked"]
fn dx12_backpressure_and_failed_batch_preserve_external_completion() {
    unsafe {
        let sdk = wgpu::Dx12BackendOptions::default()
            .with_env()
            .agility_sdk
            .expect("set Agility SDK environment");
        let _layers = libloading::Library::new(
            std::path::Path::new(&sdk.sdk_path).join("d3d12SDKLayers.dll"),
        )
        .unwrap();
        let path = std::ffi::CString::new(sdk.sdk_path).unwrap();
        let mut config = None;
        D3D12GetInterface::<ID3D12SDKConfiguration1>(&CLSID_D3D12SDKConfiguration, &mut config)
            .unwrap();
        let factory: ID3D12DeviceFactory = config
            .unwrap()
            .CreateDeviceFactory(sdk.sdk_version, PCSTR(path.as_ptr().cast()))
            .unwrap();
        let debug: ID3D12Debug = factory
            .GetConfigurationInterface(&CLSID_D3D12Debug)
            .unwrap();
        debug.EnableDebugLayer();
        let dxgi: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter = (0..32)
            .find_map(|i| {
                dxgi.EnumAdapters1(i).ok().filter(|a| {
                    String::from_utf16_lossy(&a.GetDesc1().unwrap().Description)
                        .contains("RTX 4060")
                })
            })
            .unwrap();
        let mut host = None;
        factory
            .CreateDevice::<_, ID3D12Device>(&adapter, D3D_FEATURE_LEVEL_12_0, &mut host)
            .unwrap();
        let host = host.unwrap();
        // Host debug retrieval filters are independent of storage. Reading the
        // unfiltered count would manufacture a query failure and quarantine.
        let info = host.cast::<ID3D12InfoQueue>().unwrap();
        info.AddApplicationMessage(
            D3D12_MESSAGE_SEVERITY_WARNING,
            PCSTR(c"controlled host retrieval-filter fixture".as_ptr().cast()),
        )
        .unwrap();
        let mut severities = [
            D3D12_MESSAGE_SEVERITY_ERROR,
            D3D12_MESSAGE_SEVERITY_CORRUPTION,
        ];
        let filter = D3D12_INFO_QUEUE_FILTER {
            AllowList: D3D12_INFO_QUEUE_FILTER_DESC {
                NumSeverities: 2,
                pSeverityList: severities.as_mut_ptr(),
                ..Default::default()
            },
            ..Default::default()
        };
        info.PushRetrievalFilter(&filter).unwrap();
        assert!(info.GetNumStoredMessages() > info.GetNumStoredMessagesAllowedByRetrievalFilter());
        let desc = Dx12DeviceDesc {
            next: std::ptr::null(),
            flags: 4,
            host_device: host.as_raw(),
            agility_sdk_version: sdk.sdk_version,
            agility_sdk_path: path.as_ptr(),
            max_pending_submissions: 1,
        };
        let mut error = [0i8; 4096];
        let mut device = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_device_create_dx12(&desc, &mut device, error.as_mut_ptr(), error.len()),
            0,
            "{}",
            std::ffi::CStr::from_ptr(error.as_ptr()).to_string_lossy()
        );
        let mut allocations = Vec::new();
        let mut imported = Vec::new();
        for _ in 0..4 {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer = None;
            host.CreateCommittedResource::<ID3D12Resource>(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .unwrap();
            let buffer = buffer.unwrap();
            let mut handle = std::ptr::null_mut();
            assert_eq!(
                oidnw_gpu_buffer_import_dx12(
                    device,
                    buffer.as_raw(),
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len()
                ),
                0
            );
            let mut duplicate = std::ptr::null_mut();
            assert_eq!(
                oidnw_gpu_buffer_import_dx12(
                    device,
                    buffer.as_raw(),
                    &mut duplicate,
                    error.as_mut_ptr(),
                    error.len()
                ),
                OIDNW_ERROR_INVALID_ARGUMENT
            );
            imported.push(handle);
            allocations.push(buffer);
        }
        let ready: ID3D12Fence = host.CreateFence(0, D3D12_FENCE_FLAG_SHARED).unwrap();
        let done: ID3D12Fence = host.CreateFence(0, D3D12_FENCE_FLAG_SHARED).unwrap();
        let mut ready_handle = std::ptr::null_mut();
        let mut done_handle = std::ptr::null_mut();
        assert_eq!(
            oidnw_gpu_sync_import_dx12(
                device,
                ready.as_raw(),
                &mut ready_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(
            oidnw_gpu_sync_import_dx12(
                device,
                done.as_raw(),
                &mut done_handle,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        // This test targets ownership/admission/recovery. Checked-shader timing
        // is measured separately and must never silently select unchecked here.
        super::tests::exercise_failure_contract(
            device,
            imported,
            ready_handle,
            done_handle,
            0,
            || {
                let v = done.GetCompletedValue();
                assert_ne!(v, u64::MAX);
                v
            },
            |v| ready.Signal(v).unwrap(),
        );
        ready.Signal(3).unwrap();
        assert_eq!(ready.GetCompletedValue(), 3);
        drop(allocations);
    }
}
