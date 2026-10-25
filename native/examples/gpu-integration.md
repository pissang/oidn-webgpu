# Integrating the GPU C API

Use this sequence when the renderer already owns its GPU resources. The snippets use existing host objects; the complete programs below include native device creation, shaders and command submission.

| Backend | Complete program | Native resources |
| --- | --- | --- |
| Metal | [metal_gpu.m](metal_gpu.m) | `MTLDevice`, `MTLBuffer`, `MTLSharedEvent` |
| Vulkan | [vulkan_gpu.cpp](vulkan_gpu.cpp) | Host instance, device, queue, storage buffers and timeline semaphores |
| DX12 | [dx12_gpu.cpp](dx12_gpu.cpp) | Host device, shared committed UAV buffers and shared fences; the library creates another device on the same LUID |

Build commands and backend requirements are in the [native guide](../README.md#gpu-c-api). Use matching headers and a library. Include `oidn_webgpu_metal.h`, `oidn_webgpu_vulkan.h` or `oidn_webgpu_dx12.h` for the chosen backend. Keep an error buffer such as `char error[1024]` and handle every returned status.

Images use tightly packed RGBA32F storage buffers. If the renderer uses textures, convert or copy them to these buffers on the GPU and convert the output back as needed. Those passes add time and memory beyond denoising.

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

This creates a dedicated library queue. For a shared host queue, use the [Metal queue contract](../README.md#metal-device-options).

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

Use `void lock_host_queue(void *)` and `void unlock_host_queue(void *)` callbacks. Release that lock before entering the library. The device must have been created with the [required features, extensions and queue configuration](../README.md#vulkan-device-options); importing an arbitrary existing `VkDevice` can fail.

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

The SDK directory is an absolute UTF-8 path ending in `/` or `\`. Use the host's exact SDK version. The context has its own device and queue; [DX12 sharing and resource-state requirements](../README.md#dx12-device-options) apply.

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

Use `{NULL, 0}` for unused auxiliary images. Preparation is outside the frame loop. If dimensions change, prepare another workspace; at fixed dimensions, pass the current frame's imported buffers to each submit. Tiling reduces workspace memory, while full image allocations remain required.

## 4. Submit each frame

The producer writes inputs and signals its ready value independently of the CPU submit returning. Choose increasing completion values, coordinated with any host signals to the same native object.

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

Store both `status` and `sequence`. A nonzero sequence records accepted work even when status is an error. A zero sequence with `QUEUE_FULL` permits retry after earlier work progresses; do not attach a consumer wait to that rejected signal. Follow the [execution and error contract](../README.md#objects-and-execution) for other failures.

After a successful submit, the host consumer waits on `done_value` before reading output. It does not need a CPU wait. Use the native backend's required memory scopes or resource states at both handoffs. Before reusing an image set, also account for the host consumer's completion.

For CPU reclamation, poll the returned sequence:

```c
uint32_t completed = 0;
status = oidnw_gpu_poll(device, sequence, &completed, error, sizeof error);
```

## 5. Shut down

Release the public workspace, image wrappers and synchronization wrappers, then drain accepted sequences with `oidnw_gpu_wait` before destroying the library device. Releasing public handles alone does not complete GPU work. On a timeout or uncertain submission, follow the [lifetime and recovery contract](../README.md#objects-and-execution) rather than freeing native allocations.

For Vulkan, keep all borrowed native objects and their memory alive until library device destruction succeeds. For Metal and DX12, retained library objects do not replace the host consumer's own resource lifetime requirements.

## C++17 wrapper

Use C++17 with exceptions enabled (`/EHsc` with MSVC). Include the backend's `.hpp` instead of `.h`. These headers wrap the same C ABI and do not upload or read back image pixels. Build the complete [Metal C++ program](metal_gpu_cpp.mm) with the `metal_gpu_cpp` CMake target; link other C++ hosts to `oidn_webgpu::cpp`.

Choose a platform entry point, using the C descriptions from step 1:

| Backend | C++ context | Images | Synchronization |
| --- | --- | --- | --- |
| Metal | `oidnw::metal::import_device(desc)` | `import_buffer(context, native, bytes)` | `import_sync(context, native)` |
| Vulkan | `oidnw::vulkan::import_device(desc)` | `import_buffer(context, native, create_info)` | `import_sync(context, native, create_info)` |
| DX12 | `oidnw::dx12::create_device(desc)` | `import_buffer(context, native)` | `import_sync(context, native)` |

The resource functions use the same namespace as the context function. Pass `.get()` from the returned buffer/sync wrappers when building the common C image and timeline-point structures. The common flow then uses the same C++ types on every platform:

```cpp
auto model = oidnw::gpu::load_model(weights, weights_size);
auto options = oidnw::gpu::filter_options();
options.max_tile_size = 128;
auto filter = oidnw::gpu::make_filter(context, model, options);
auto prepared = oidnw::gpu::prepare(filter, width, height, images);

auto submitted = prepared.submit(images, wait, signal, input_scale);
```

Initialization failures throw `oidnw::gpu::error`, whose `code` is the native status. For multiple waits or signals, pass an `oidnw_gpu_submit_desc` to the same `submit` method. Submission never throws: inspect `submitted.code`, `submitted.sequence`, `submitted.accepted()` and `submitted.message`. A nonzero error status can still have an accepted sequence. `context.poll(sequence)` and `context.wait(sequence, timeout_ns)` return both status and a `completed` boolean. GPU synchronization remains the host's responsibility.

The C API's same-handle concurrency rules also apply to these wrappers. All wrappers are move-only. Buffer, sync, filter and workspace wrappers hold the context alive even if its original C++ device owner has been moved or destroyed. Keep native Vulkan objects and queue callback userdata alive as well. An optional second argument to each context factory supplies a lifetime token:

```cpp
oidnw::gpu::device_options ownership;
ownership.host_lifetime = native_owner;
auto context = oidnw::vulkan::import_device(desc, ownership);
```

Here `native_owner` is a `std::shared_ptr` to a separate owner of the native instance/device and queue lock. Keep the C++ device and children outside that object.

Release child wrappers before calling `context.close(timeout_ns)`. A busy or timed-out close leaves ownership intact. Last-owner destruction waits for accepted submissions without a deadline, so it can block. Producers must signal independently of this cleanup. For C submissions made through borrowed `.get()` handles, register every accepted sequence with `context.track(sequence)` and release the raw public children before closing.

After releasing child wrappers, if shutdown needs host recovery, `context.release()` returns an `oidnw::gpu::detached_device` containing the native handle and host-lifetime token. Retain both until safe C API destruction succeeds. `context.close_quiescent()` delegates to the native recovery contract and requires host proof that GPU access has ceased. Default destructor cleanup failure terminates. A custom `device_options::on_cleanup_failure` callback can take the detached context and token instead; the callback must be `noexcept` and keep them alive for host recovery.

## Vulkan preflight

Get the library's instance extensions before `vkCreateInstance`, then merge any extensions the host needs:

```cpp
auto names = oidnw::vulkan::instance_extensions(
    host_get_instance_proc_addr, VK_API_VERSION_1_2);
host_instance_create_info.enabledExtensionCount = uint32_t(names.size());
host_instance_create_info.ppEnabledExtensionNames = names.data();
```

After selecting a physical device, query the logical-device requirements:

```cpp
oidnw_gpu_vulkan_requirements_desc query;
oidnw_gpu_vulkan_requirements_desc_default(&query);
query.flags = import_options.flags;
query.get_instance_proc_addr = host_get_instance_proc_addr;
query.instance = host_instance;
query.physical_device = host_physical_device;
query.instance_create_info = &host_instance_create_info;
auto requirements = oidnw::vulkan::query_requirements(query);
auto create_info = oidnw::vulkan::device_create_info(requirements);
create_info.queueCreateInfoCount = uint32_t(queue_infos.size());
create_info.pQueueCreateInfos = queue_infos.data();
```

Use this declaration for `vkCreateDevice`, then supply it as `import_options.device_create_info` when importing the resulting device/queue. Queue declarations must provide unprotected graphics/compute queues. Keep `requirements`, the instance declaration and host queue declarations alive through import. No logical device or GPU work is created by the queries.

The returned extension array, core features and feature-chain nodes are read-only and owned by `requirements`. The Vulkan import restrictions still apply to any host additions. For additional host requirements, merge extension names and copy/merge feature bits without duplicating feature structures. Aggregated Vulkan 1.1/1.2/1.3 structures are unsupported. The complete [Vulkan host](vulkan_gpu.cpp) uses this preflight sequence.
