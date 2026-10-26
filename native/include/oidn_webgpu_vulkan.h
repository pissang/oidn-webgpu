#ifndef OIDN_WEBGPU_VULKAN_H
#define OIDN_WEBGPU_VULKAN_H
#include "oidn_webgpu_gpu.h"
#include <vulkan/vulkan.h>
#ifdef __cplusplus
extern "C" {
#endif

#define OIDNW_GPU_VULKAN_FP32 1u
#define OIDNW_GPU_VULKAN_NO_SUBGROUPS 2u
/* Cooperative matrices are enabled by default when supported. This flag disables
 * wgpu's experimental matrix feature; matrix behavior is outside semver guarantees. */
#define OIDNW_GPU_VULKAN_NO_COOPERATIVE_MATRIX 4u
typedef struct oidnw_gpu_vulkan_requirements oidnw_gpu_vulkan_requirements;
typedef struct oidnw_gpu_vulkan_requirements_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  uint32_t flags;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  const VkInstanceCreateInfo *instance_create_info;
} oidnw_gpu_vulkan_requirements_desc;

/* Query before vkCreateInstance. With names=NULL, writes the required count.
 * Otherwise *count is the array capacity; always writes the required count.
 * A short array returns INVALID_ARGUMENT without writing names. Returned names
 * are immutable library strings valid until the library is unloaded.
 * Does not add host-requested layers/extensions such as debug utilities. */
int32_t oidnw_gpu_vulkan_instance_extensions(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
  uint32_t api_version, const char **names, size_t *count, char *error, size_t capacity);
void oidnw_gpu_vulkan_requirements_desc_default(oidnw_gpu_vulkan_requirements_desc *desc);
/* Query after instance/physical-device selection, before vkCreateDevice.
 * Supported cooperative-matrix extensions/features are included by default;
 * NO_COOPERATIVE_MATRIX omits them. Import uses matrices only when the host enabled
 * both VK_KHR_cooperative_matrix and cooperativeMatrix. flags must match import.
 * No logical device or GPU work is created.
 * Host loader, instance and creation declarations must be live during this call.
 * Queries return UNAVAILABLE outside Windows/Linux. */
int32_t oidnw_gpu_vulkan_requirements_create(const oidnw_gpu_vulkan_requirements_desc *desc,
  oidnw_gpu_vulkan_requirements **requirements, char *error, size_t capacity);
/* Writes a base VkDeviceCreateInfo with required extensions, core features and
 * the canonical pNext chain; host supplies queue declarations before creation.
 * All pointed-to data is read-only and lives until requirements_destroy. Keep
 * this owner alive through library import, which reads the original declaration.
 * Host additions require merging extension names and feature bits, without
 * duplicate feature structs; aggregated Vulkan11/12/13 structs remain unsupported.
 * enabled_features is optional and receives OIDNW_FEATURE_* bits. On failure,
 * each non-NULL, aligned output is zeroed, including when another output is
 * invalid. A NULL create_info is rejected. */
int32_t oidnw_gpu_vulkan_requirements_get(const oidnw_gpu_vulkan_requirements *requirements,
  VkDeviceCreateInfo *create_info, uint32_t *enabled_features, char *error, size_t capacity);
void oidnw_gpu_vulkan_requirements_destroy(oidnw_gpu_vulkan_requirements *requirements);

typedef struct oidnw_gpu_vulkan_device_desc {
  const void *next; /* NULL; non-NULL extensions are rejected. */
  uint32_t flags;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;
  uint32_t queue_index;
  const VkInstanceCreateInfo *instance_create_info;
  const VkDeviceCreateInfo *device_create_info;
  oidnw_gpu_queue_lock_fn lock;
  oidnw_gpu_queue_lock_fn unlock;
  void *user_data;
  uint32_t max_pending_submissions;
} oidnw_gpu_vulkan_device_desc;

void oidnw_gpu_vulkan_device_desc_default(oidnw_gpu_vulkan_device_desc *desc);
/* Route A: borrow the host instance/device/queue, without destroying them.
 * All native objects and buffer memory must outlive successful library device
 * destruction, including after public buffer/sync handles have been released.
 * Supply the original create declarations and live arrays/pNext chains. Import
 * validates every HAL-required extension and feature for wgpu-hal 30.0.1, not
 * just denoising shader features. Use individual feature structs; aggregated
 * Vulkan11/12/13 feature structs are rejected. Vulkan >=1.2 is required.
 * Supply core features through pEnabledFeatures; only the recognized canonical
 * HAL feature chain is accepted (including no VkPhysicalDeviceFeatures2 wrapper).
 * The queue must be unprotected and support GRAPHICS|COMPUTE. Host and library
 * must use the same paired lock/unlock callbacks for ALL access to this queue.
 * Callbacks must not reenter, throw or unwind; user_data outlives destruction.
 * Do not call the library while holding that host queue lock.
 * No external-memory fallback or separate-device route is implemented. */
int32_t oidnw_gpu_device_import_vulkan(const oidnw_gpu_vulkan_device_desc *desc,
  oidnw_gpu_device **device, char *error, size_t capacity);
/* Borrow a VkBuffer from the imported VkDevice, fully bound, non-sparse and
 * unprotected, with STORAGE_BUFFER usage.
 * create_info must describe its actual creation, including size and sharing.
 * Exclusive buffers must be owned by the imported queue family before use;
 * concurrent buffers must include it. Host handles queue-family transfers.
 * No overlapping memory aliases. Import a buffer only once per context while
 * any library reference survives. Keep native allocations alive until device
 * destruction. No CPU pixel upload/readback is performed by this interface. */
int32_t oidnw_gpu_buffer_import_vulkan(oidnw_gpu_device *device, VkBuffer buffer,
  const VkBufferCreateInfo *create_info, oidnw_gpu_buffer **result, char *error, size_t capacity);
/* Borrow a timeline semaphore from the imported VkDevice. The original create
 * info must contain VkSemaphoreTypeCreateInfo(TIMELINE). Binary semaphores are
 * rejected. Host must provide memory dependencies through waits/signals at
 * ALL_COMMANDS (or an equivalently covering scope); image uses are compute
 * storage reads/writes. Producers must signal independently of submit return.
 * Each semaphore may appear once in waits and once in signals per submit. */
int32_t oidnw_gpu_sync_import_vulkan(oidnw_gpu_device *device, VkSemaphore semaphore,
  const VkSemaphoreCreateInfo *create_info, oidnw_gpu_sync **result, char *error, size_t capacity);
/* Pinned HAL relay constraint: while an earlier submission has unresolved
 * external waits, a further submit returns QUEUE_FULL with sequence=0. This
 * conservative admission rule also applies to GPU producers whose future
 * signals have been submitted but are not yet observable. Poll/wait for the
 * accepted batch before retrying; never make a producer depend on that retry.
 * Deep chains behind unresolved CPU timeline signals are not supported. */
#ifdef __cplusplus
}
#endif
#endif
