# Integrating the GPU C API

This walkthrough adapts an existing renderer that already owns its GPU device and buffers. It uses the plain C API so every call is visible; C++ hosts should start from the [C++ Vulkan example](../README.md#example) and the [C++17 wrapper](#c17-wrapper) section, which follow the same steps.

| Backend | Complete program | Native resources |
| --- | --- | --- |
| Metal | [metal_gpu_cpp.mm](metal_gpu_cpp.mm) (C++), [metal_gpu.m](metal_gpu.m) (C) | `MTLDevice`, `MTLBuffer`, `MTLSharedEvent` |
| Vulkan | [vulkan_gpu.cpp](vulkan_gpu.cpp) | Host instance, device, queue, storage buffers and timeline semaphores |
| DX12 | [dx12_gpu.cpp](dx12_gpu.cpp) | Host device, shared committed UAV buffers and shared fences; the library creates another device on the same LUID |
| CUDA / Windows DX12 | [cuda_dx12_gpu.cu](cuda_dx12_gpu.cu) | CUDA pointers and streams; the optional [adapter](#windows-cuda-adapter) owns DX12 bridge resources |

Build commands and per-platform requirements are in the [native guide](../README.md#gpu-c-api). Include `oidn_webgpu_metal.h`, `oidn_webgpu_vulkan.h` or `oidn_webgpu_dx12.h`, keep an error buffer such as `char error[1024]`, and check every returned status. Images are RGBA32F storage buffers ([image contract](../README.md#image-contract)).

## 1. Create the library context

### Metal

```objective-c
oidnw_gpu_metal_device_desc desc;
oidnw_gpu_metal_device_desc_default(&desc);
desc.device = (__bridge void *)host_device;
oidnw_gpu_device *device = NULL;
int32_t status = oidnw_gpu_device_import_metal(
    &desc, &device, error, sizeof error);
```

This creates a dedicated library queue. For a shared host queue, use the [Metal queue contract](../README.md#metal).

### Vulkan

```c
oidnw_gpu_vulkan_device_desc desc;
oidnw_gpu_vulkan_device_desc_default(&desc);
desc.get_instance_proc_addr = host_get_instance_proc_addr;
desc.instance = host_instance;
desc.physical_device = host_physical_device;
desc.device = host_device;
desc.queue = host_queue;
desc.queue_family_index = host_queue_family;
desc.queue_index = host_queue_index;
desc.instance_create_info = &host_instance_create_info;
desc.device_create_info = &host_device_create_info;
desc.lock = lock_host_queue;
desc.unlock = unlock_host_queue;
desc.user_data = host_queue_mutex;
oidnw_gpu_device *device = NULL;
int32_t status = oidnw_gpu_device_import_vulkan(
    &desc, &device, error, sizeof error);
```

Use `void lock_host_queue(void *)` and `void unlock_host_queue(void *)` callbacks. Release that lock before entering the library. The device must have been created with the [required features, extensions and queue configuration](../README.md#vulkan); importing an arbitrary existing `VkDevice` can fail.

### DX12

```c
oidnw_gpu_dx12_device_desc desc;
oidnw_gpu_dx12_device_desc_default(&desc);
desc.host_device = host_device;
desc.agility_sdk_version = host_sdk_version;
desc.agility_sdk_path = host_sdk_directory;
oidnw_gpu_device *device = NULL;
int32_t status = oidnw_gpu_device_create_dx12(
    &desc, &device, error, sizeof error);
```

The SDK directory is an absolute UTF-8 path ending in `/` or `\`. Use the host's exact SDK version. The context has its own device and queue; [DX12 sharing and resource-state requirements](../README.md#dx12) apply.

## 2. Import images and synchronization

Each image uses a host buffer. For example, import the color buffer with one of these calls:

```objective-c
status = oidnw_gpu_buffer_import_metal(device, (__bridge void *)color_buffer,
    color_buffer.length, &color, error, sizeof error);
```

```c
status = oidnw_gpu_buffer_import_vulkan(device, color_buffer,
    &color_buffer_create_info, &color, error, sizeof error);
```

```c
status = oidnw_gpu_buffer_import_dx12(device, color_resource,
    &color, error, sizeof error);
```

Here `color` is an `oidnw_gpu_buffer *` initialized to NULL. Import albedo, normal and a separate output buffer in the same way. Only import the auxiliary images the selected model uses. Keep wrappers for reuse across frames; importing the same retained native object twice is rejected.

Import the producer's ready object and the denoiser's completion object as `oidnw_gpu_sync *ready` and `*done`:

| Backend | Import function | Extra declaration |
| --- | --- | --- |
| Metal | `oidnw_gpu_sync_import_metal` | Native `MTLSharedEvent` pointer |
| Vulkan | `oidnw_gpu_sync_import_vulkan` | Original `VkSemaphoreCreateInfo`, including timeline type |
| DX12 | `oidnw_gpu_sync_import_dx12` | Host shared fence pointer |

The [image contract](../README.md#image-contract) specifies layout and aliasing. Query `oidnw_gpu_get_limits` before choosing offsets and sizes.

## 3. Prepare once for an image size

This helper creates a fixed-size workspace and releases temporary model/filter handles on every return path. The caller owns the returned workspace.

```c
#include "oidn_webgpu_gpu.h"

int32_t prepare_denoiser(oidnw_gpu_device *device,
    const void *weights, size_t weights_size, uint32_t width, uint32_t height,
    const oidnw_gpu_images *images, oidnw_gpu_prepared **prepared,
    char *error, size_t capacity) {
  if (!prepared) return OIDNW_ERROR_INVALID_ARGUMENT;
  *prepared = NULL;
  oidnw_gpu_model *model = NULL;
  oidnw_gpu_filter *filter = NULL;
  int32_t status = oidnw_gpu_model_create(weights, weights_size,
      &model, error, capacity);
  if (status != OIDNW_OK) goto cleanup;

  oidnw_gpu_filter_desc options;
  oidnw_gpu_filter_desc_default(&options);
  options.max_tile_size = 128;
  status = oidnw_gpu_filter_create(device, model, &options,
      &filter, error, capacity);
  if (status != OIDNW_OK) goto cleanup;
  status = oidnw_gpu_prepare(filter, width, height, images,
      prepared, error, capacity);

cleanup:
  oidnw_gpu_filter_destroy(filter);
  oidnw_gpu_model_destroy(model);
  return status;
}
```

Pass image bindings in this order:

```c
oidnw_gpu_images images = {
    {color, offset}, {albedo, offset}, {normal, offset}, {output, offset}
};
```

Use `{NULL, 0}` for unused auxiliary images. Preparation is outside the frame loop. If dimensions change, prepare another workspace; at fixed dimensions, pass the current frame's imported buffers to each submit.

## 4. Submit each frame

Choose `ready_value` and `done_value` so that each native sync object's values increase across the library's and your own signals. The [frame loop rules](../README.md#frame-loop) cover producer ordering, backpressure and reclamation.

```c
oidnw_gpu_timeline_point wait = {ready, ready_value};
oidnw_gpu_timeline_point signal = {done, done_value};
oidnw_gpu_submit_desc submit;
oidnw_gpu_submit_desc_default(&submit);
submit.images = images;
submit.input_scale = input_scale;
submit.waits = &wait;
submit.wait_count = 1;
submit.signals = &signal;
submit.signal_count = 1;
uint64_t sequence = 0;
status = oidnw_gpu_submit(prepared, &submit, &sequence,
    error, sizeof error);
```

Keep both `status` and `sequence`; [Errors and recovery](../README.md#errors-and-recovery) lists what each combination means. On success, your consumer waits on `done_value` on the GPU before reading the output. Use the backend's required memory scopes or resource states at both handoffs.

Poll the returned sequence from time to time so completed work is reclaimed:

```c
uint32_t completed = 0;
status = oidnw_gpu_poll(device, sequence, &completed, error, sizeof error);
```

## 5. Shut down

```c
oidnw_gpu_prepared_destroy(prepared);
oidnw_gpu_buffer_destroy(color);   /* and albedo, normal, output */
oidnw_gpu_sync_destroy(ready);
oidnw_gpu_sync_destroy(done);
uint32_t completed = 0;
status = oidnw_gpu_wait(device, last_sequence, UINT64_MAX, &completed, error, sizeof error);
if (status == OIDNW_OK && completed) status = oidnw_gpu_device_destroy(device, error, sizeof error);
```

Free your native buffers, syncs and device only after `oidnw_gpu_device_destroy` succeeds. The order, `BUSY` retries and recovery after errors are in [Lifetime and threads](../README.md#lifetime-and-threads) and [Errors and recovery](../README.md#errors-and-recovery).

## C++17 wrapper

Use C++17 with exceptions enabled (`/EHsc` with MSVC). Include the backend's `.hpp` instead of `.h`. These headers wrap the same C ABI and do not upload or read back image pixels. Build the complete [Metal C++ program](metal_gpu_cpp.mm) with the `metal_gpu_cpp` CMake target; link other C++ hosts to `oidn_webgpu::cpp`.

The Windows `vulkan_gpu` and `dx12_gpu` programs also run the common C++ model/filter/workspace/buffer/sync flow through their platform helpers. Their `gpu_cpp_lifecycle.hpp` helper uses a separate native resource owner and a gated producer for explicit timeout and retry shutdown. GPU-validation mode keeps the existing diagnostic protocol and skips this additional gated lifecycle flow. The CMake target enables C++17 and MSVC exception handling.

Create the context and imports with the C descriptions from step 1 and the functions in the [platform import table](../README.md#platform-imports). The common flow then uses the same C++ types on every platform:

```cpp
auto model = oidnw::gpu::load_model(weights, weights_size);
auto options = oidnw::gpu::filter_options();
options.max_tile_size = 128;
auto filter = oidnw::gpu::make_filter(context, model, options);
auto prepared = oidnw::gpu::prepare(filter, width, height, images);

auto submitted = prepared.submit(images, wait, signal, input_scale);
```

For multiple waits or signals, pass an `oidnw_gpu_submit_desc` (from `oidnw::gpu::submit_options()`) to the same `submit` method. The returned `submission` has `code`, `message`, `sequence` and `accepted()`; `context.poll(sequence)` and `context.wait(sequence, timeout_ns)` return a `completion` with `code` and `completed`.

All wrappers are move-only, and buffer, sync, filter and workspace wrappers keep the context alive even after the C++ device owner is moved or destroyed. An optional second argument to each context factory supplies a lifetime token for your native objects:

```cpp
oidnw::gpu::device_options ownership;
ownership.host_lifetime = native_owner;
auto context = oidnw::vulkan::import_device(desc, ownership);
```

Here `native_owner` is a `std::shared_ptr` to a separate owner of the native instance/device and queue lock. Keep the C++ device and children outside that object. The Vulkan `import_device(handles, instance_info, device_info, lock)` overload retains its `queue_lock` the same way, together with any `host_lifetime` you pass.

Release child wrappers before calling `context.close(timeout_ns)`; a busy or timed-out close leaves ownership intact. Producers must signal independently of cleanup. For C submissions made through borrowed `.get()` handles, register every accepted sequence with `context.track(sequence)` and release the raw public children before closing.

After releasing child wrappers, if shutdown needs host recovery, `context.release()` returns an `oidnw::gpu::detached_device` containing the native handle and host-lifetime token. Retain both until safe C API destruction succeeds. `context.close_quiescent()` delegates to the native recovery contract and requires host proof that GPU access has ceased. Default destructor cleanup failure terminates. A custom `device_options::on_cleanup_failure` callback can take the detached context and token instead; the callback must be `noexcept` and keep them alive for host recovery.

GPU filter descriptions also select `kernel` (default AUTO) and `quality` (default HIGH). Metal/Vulkan devices enable supported cooperative matrices by default; use `OIDNW_GPU_METAL_NO_COOPERATIVE_MATRIX` or `OIDNW_GPU_VULKAN_NO_COOPERATIVE_MATRIX` to disable them. Vulkan requirements queries include the matrix extension and feature only when supported and not disabled; use the same flags for import, which uses matrices only if the host enabled them. AUTO falls back to implicit GEMM, while an explicit SUBGROUP_MATRIX request without the enabled feature returns UNAVAILABLE. See [filter options and hardware coverage](../README.md#filter-options).

Initialize every descriptor with its default function and leave `next` NULL. Query immutable workspace memory and kernel information with `oidnw_gpu_prepared_get_info` (C++: `workspace.info()`), initializing the output to `{0}`. Queries may overlap submit on the same workspace while it remains alive.

## Windows CUDA adapter

[`oidn_webgpu_cuda.hpp`](../include/oidn_webgpu_cuda.hpp) and `oidn_webgpu::cuda` provide an optional C++17 adapter for CUDA renderers. The adapter owns the DX12 devices, shared committed buffers, external-memory mappings, fences, NT handles and Agility SDK factory. Applications supply CUDA streams and image descriptions. Denoising uses the existing DX12 backend and C ABI 3.0.0; there is no CUDA denoising backend. Use matching adapter headers, adapter library and core DLL.

| Entry point | Input/output storage | Bridge image copies |
| --- | --- | --- |
| `enqueue_copy(images, stream, input_scale)` | Caller-owned CUDA device allocations | Three GPU input copies and one GPU output copy; no CPU image transfer |
| `acquire()`, `enqueue_shared(lease, stream, input_scale)`, `release(lease, consumer_stream)` | Adapter-owned shared image set | Zero; CUDA and DX12 access the same allocations |

Both paths use RGBA float32, including FP16 denoising. Native normals use signed [-1, 1] components. This adapter supports Windows CUDA/DX12 only; CUDA/Vulkan is outside its scope.

### Build and deploy

The default native package ships the public header, adapter implementation in `examples/cuda-adapter/`, and [`cuda_dx12_gpu.cu`](cuda_dx12_gpu.cu). It does not build or load CUDA. Enabling the adapter requires 64-bit Windows, CUDA Toolkit 12.5 or newer, and a CUDA-supported NVIDIA driver for that Toolkit. See NVIDIA's [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) for driver requirements and [external resource interoperability](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-c-programming-guide/index.html#direct3d-12-interoperability) for the D3D12 resource contract.

Build a native package first, then run from the repository root in an x64 VS developer PowerShell. Use a CUDA-supported host compiler and install Ninja, or select another CUDA-capable CMake generator.

```powershell
npm run native:build
$env:WGPU_DX12_AGILITY_SDK_PATH = 'C:/SDKs/Agility/build/native/bin/x64/'
$env:WGPU_DX12_AGILITY_SDK_VERSION = '619'
cmake -S .native-build/package/examples -B .native-build/cuda-examples -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" `
  -DOIDNW_BUILD_CUDA_DX12_GPU_EXAMPLE=ON
cmake --build .native-build/cuda-examples --target cuda_dx12_gpu
.native-build/cuda-examples/cuda_dx12_gpu.exe weights/rt_hdr_alb_nrm_small.tza 1000 16 0 0 0
.native-build/cuda-examples/cuda_dx12_gpu.exe weights/rt_hdr_alb_nrm_small.tza 1000 16 0 0 1
```

The last argument selects shared images (`0`) or ordinary pointers (`1`). Further positional arguments select runtime checks (default `1`), DX12 GPU validation (default `0`) and diagnostic mode (default `0`). Diagnostic mode permits already-completed delays under instrumentation and reports whether pending states were observed; it keeps the same GPU image comparison. The full usage is printed when no model is supplied. The example uses seven content/exposure combinations and three resource sets. Both modes compare with serial shared-image references, so the ordinary-pointer path also checks its pitched copies against the path without bridge copies. Reference storage and CUDA consumption stay on the GPU; only four integer counters reach the CPU.

To build only the adapter, without compiling CUDA kernels, configure `native/cuda` in a source checkout or `examples/cuda-adapter` in an extracted package:

```powershell
cmake -S .native-build/package/examples/cuda-adapter -B .native-build/cuda-adapter `
  -DCMAKE_PREFIX_PATH="$PWD/.native-build/package" -DOIDNW_BUILD_CUDA_ADAPTER=ON
cmake --build .native-build/cuda-adapter --config Release
cmake --install .native-build/cuda-adapter --config Release --prefix .native-build/package
```

Consumers of this installation call `find_package(oidn_webgpu_cuda CONFIG REQUIRED)` and link `oidn_webgpu::cuda`. Alternatively, set `OIDNW_BUILD_CUDA_ADAPTER=ON` and use `add_subdirectory` on the shipped adapter source. The target compiles as C++17 with the DLL CRT and links the CUDA runtime/driver and core DLL import library. Compile consumers with matching C++ headers, compiler ABI and CRT. Disabling the option requires neither CUDA nor Agility headers, on Windows or other platforms.

Deploy `oidn_webgpu.dll` beside the application and the CUDA runtime DLL selected by the Toolkit through the application's DLL search path. For FP16, also deploy matching DXC `dxcompiler.dll`/`dxil.dll`; construction rejects device/compiler combinations that do not expose shader-f16. Supply `D3D12Core.dll` from the matching Agility SDK in the configured directory; debug mode also requires its `d3d12SDKLayers.dll`. Adapter creation loads the SDK and initializes both DX12 devices internally. It does not download or redistribute these dependencies. No DX12 initialization code is needed in the application.

### Configuration and context

```cpp
#include <oidn_webgpu_cuda.hpp>
namespace bridge = oidnw::cuda;

cudaSetDevice(renderer_device); // Establish the renderer's primary context first.
bridge::options options;
options.width = width;
options.height = height;
bridge::adapter denoiser(options, tza_bytes, tza_size);
```

Construction and metadata/view getters throw on failure. Enqueue, acquire, release, completion and shutdown operations return `status`, or an object containing `result`. Check every result. `status` converts to true on success. The adapter matches the current CUDA device to DX12 by the complete GPU LUID and rejects linked-node adapters.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `width`, `height` | `uint32_t` | `0` | Required image dimensions, 1..16384; device and workspace limits also apply |
| `max_tile_size` | `uint32_t` | `0` | Existing GPU filter tiling option |
| `slots` | `uint32_t` | `3` | Fixed pool capacity, 1..64 image sets |
| `hdr` | `bool` | `true` | `false` for LDR weights |
| `transfer` | `uint32_t` | `OIDNW_TRANSFER_PU` | `OIDNW_TRANSFER_LOG` for RTLightmap HDR weights |
| `fp32` | `bool` | `false` | Force FP32; otherwise require FP16 |
| `runtime_checks` | `bool` | `true` | Existing backend shader checks |
| `debug` | `bool` | `false` | Enable the DX12 debug layer on both devices |
| `gpu_validation` | `bool` | `false` | Enable DX12 GPU validation on both devices; requires `debug` |
| `agility_sdk_path` | `std::string` | empty | Absolute UTF-8 SDK directory with trailing separator; empty uses `WGPU_DX12_AGILITY_SDK_PATH` |
| `agility_sdk_version` | `uint32_t` | `0` | Matching SDK version; zero uses `WGPU_DX12_AGILITY_SDK_VERSION` |

- Only the current device's primary CUDA context is supported. Establish it before construction and keep it current for calls. Ordinary operations never switch it. Destructors may temporarily push the retained context and restore the previous context.
- Supply explicit CUDA streams from that context. Legacy/default and per-thread default stream sentinels are rejected. CUDA Graph capture is detected and rejected.
- Serialize host calls to an adapter, its tasks and its image leases. Overlapping calls return `busy`; destructors serialize cleanup. Multiple GPU streams are supported within this host-call rule.
- Device reset, context destruction and unsynchronized writes from other streams are outside the contract. Do not call adapter operations or destructors from CUDA host callbacks.

### Ordinary CUDA pointers

Each `image` contains `data`, `bytes`, `offset`, `row_stride`, `width`, `height` and `pixel_format`. `data` is the allocation base or a pointer within that allocation; `bytes` is the available range from `data`. Offsets and row strides are in bytes, divisible by 16. `row_stride` must cover `width * 16` bytes. The last image row must fit inside `bytes`. The only format is `format::rgba32f`. Output must not overlap any input.

```cpp
bridge::images images{color_view, albedo_view, normal_view, output_view};
auto task = denoiser.enqueue_copy(images, renderer_stream, input_scale);
if (!task.result) {
  // Keep an accepted task and its allocations for drain, even on an error.
  handle_error(task);
} else {
  consume<<<grid, block, 0, renderer_stream>>>(output_pointer);
}
```

Supported pointers refer to same-device, primary-context device allocations with a queryable allocation range, such as `cudaMalloc`/`cuMemAlloc`. Managed, host, peer and foreign-context memory are rejected. CUDA memory-pool allocations and arbitrary imported allocations are outside this first version's supported pointer contract. There is no CPU fallback or automatic format conversion; convert other renderer formats in a CUDA kernel before enqueue.

The supplied stream must already contain input preparation. The adapter queues pitched D2D copies into shared buffers, a producer-ready signal, DX12 denoising, a CUDA wait and the D2D output copy. On success, later kernels in that stream can read output. Keep caller allocations and the stream valid until task completion; keep output valid until application consumers complete. This path incurs four image copies per execution and is not zero-copy.

### Shared images

```cpp
auto lease = denoiser.acquire();
if (!lease.result) handle_backpressure(lease.result);
auto views = lease.views();
produce<<<grid, block, 0, renderer_stream>>>(
    static_cast<char *>(views.color.data) + views.color.offset);
auto task = denoiser.enqueue_shared(lease, renderer_stream, input_scale);
if (!task.result) handle_error(task);
consume<<<grid, block, 0, renderer_stream>>>(
    static_cast<char *>(views.output.data) + views.output.offset);
auto returned = denoiser.release(lease, renderer_stream);
```

Check `returned` before dropping the lease. Fill albedo and normal too when required by the model. One acquisition permits one enqueue. Views have fixed dimensions, packed RGBA32F rows and offset 256. Each image has its own committed DEFAULT, SHARED, ROW_MAJOR UAV buffer, created in COMMON. CUDA imports use `D3D12Resource`, the dedicated flag and the DX12 allocation size; the mapped range and view length use the logical buffer size. The library independently imports the same resource. CUDA does not change DX12 resource state; the DX12 backend returns resources to COMMON. Fences order access and do not transition states.

The move-only lease retains the allocation owner. Copies of its views and bare device pointers do not. Images from another adapter are rejected, even on the same CUDA device. Shared allocation/import failure is an error; this entry point never falls back to D2D copies.

For a consumer in another stream, call `task.depend(consumer_stream)` before queueing its kernels, then `release(lease, consumer_stream)` after the last consumer. Release records retirement after those kernels and invalidates the lease. A slot is reusable only after retirement and task-handle release. A completed denoising task alone does not permit shared-slot reuse. An unreleased lease's destructor synchronizes the retained CUDA context to cover unknown consumers; explicit stream release avoids that shutdown wait. Stop submitting work that uses its pointers before either form of release.

### Completion, backpressure and shutdown

`resize(width, height)` changes dimensions while reusing both DX12 devices, the CUDA context, model and shared weights. It rebuilds the workspace and shared image pool, so it can compile shaders and temporarily hold both pools. Release all task handles (including completed tasks) and shared leases, then finish recorded consumer retirement; otherwise it returns `busy`. Invalid dimensions or allocation/preparation failure preserve the previous pool. Previously copied views and bare pointers become invalid after a successful size change. Filter options stay fixed.

Tasks and image leases are move-only. Tasks expose `accepted()`, `sequence()`, `phase()`, `query(completed)`, `wait(timeout_ns)` and `depend(stream)`. Completion covers the ordinary-pointer output copy. `depend` inserts a CUDA event dependency without a CPU wait. Task phases distinguish `not_accepted`, `accepted`, `accepted_error` and `completed`; `result` retains any enqueue error after completion. `query` can return an error together with `completed=true` for a finished failed task; inspect both outputs.

- Invalid inputs and pool exhaustion return an unaccepted object. `busy` means the caller should retire earlier work and retry; the pool never grows. Keep task handles only as long as needed, because they pin their slot and completion event.
- An accepted task can have sequence zero if CUDA bridge work was queued before DX12 rejected the submission. A nonzero sequence always remains tracked, including accepted DLL errors. Keep the task, stream and caller allocations for drain. For a failed shared task, wait until `query` confirms completion before returning its lease; release returns `busy` while its work remains pending and preserves ownership on uncertain completion. Subsequent enqueues stop after a bridge/backend failure.
- CUDA waits are issued only after the corresponding successful DX12 submission has issued its signal. Failed submissions do not place a wait for an unissued signal in the application stream. An error result does not authorize consuming output.
- Each slot has separate producer-ready and denoise-done fences. Each fence has one signal producer and increasing values. Events protect CUDA consumer retirement; fence values are never reused for another outstanding ownership cycle.
- `drain(timeout_ns)` waits for accepted work and recorded retirement. It returns `busy` for held shared leases. `close(timeout_ns)` additionally requires released task handles and closes the core context. Nanosecond timeout zero queries once; `UINT64_MAX` waits indefinitely. Timeout or busy preserves all resources, so release/drain and retry close. Successful close is idempotent.
- Adapter, task and lease owners retain their common context. Destroying the outer adapter does not invalidate surviving children. Last-owner cleanup drains accepted work and can block. Failed cleanup terminates instead of freeing resources whose GPU use is uncertain. Real device-loss recovery is not implemented.

Pool memory includes four shared image allocations per slot, CUDA mappings/events, per-slot fences, backend weights and the prepared tiled workspace. Allocation size can exceed logical image size because of DX12 heap alignment. Ordinary mode also needs the caller's input/output allocations. `stats()` reports shared allocation bytes and bridge copy bytes/enqueues; it excludes model/workspace memory and application copies. `info()` reports the loaded core DLL path, C ABI version, SDK version, compiler and immutable `workspace` information described in the [filter guide](../README.md#filter-options).

## Vulkan preflight

The [C++ Vulkan example](../README.md#example) shows the complete preflight: `instance_extensions` before `vkCreateInstance`, `query_requirements` and `merge_requirements` before `vkCreateDevice`, then import with the same create infos. This section covers what the example leaves out.

- `query.flags` must equal the later import's `flags` (for example `OIDNW_GPU_VULKAN_FP32`). Queries include supported cooperative-matrix requirements by default; `OIDNW_GPU_VULKAN_NO_COOPERATIVE_MATRIX` omits them. Import uses the feature only when the host enabled both the extension and feature. For fallback and hardware coverage, see the [Vulkan guide](../README.md#vulkan).
- The queries create no logical device and no GPU work. Keep the instance declaration, the merged device declaration and your queue declarations alive through import.
- Without your own extensions or features, `device_create_info(requirements)` returns the library's declaration directly. Its extension array, core features and feature-chain nodes are read-only and owned by `requirements`, so keep `requirements` alive through import in that case.
- From C, merge by hand: combine extension names and feature bits without duplicating feature structures. The [Vulkan import rules](../README.md#vulkan) apply either way.
- `oidnw_gpu_vulkan_requirements_get` rejects a NULL declaration output. On failure it zeroes every non-NULL, aligned declaration and feature-bit output; discard previously borrowed pointers when replacing a failed query result.

The complete [Vulkan host](vulkan_gpu.cpp) also exercises the C entry points `oidnw_gpu_vulkan_instance_extensions` and `oidnw_gpu_vulkan_requirements_create/get`.
