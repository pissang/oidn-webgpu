/* A GPU-independent shared-library/header ABI smoke test for CI packages. */
#include <stdio.h>
#include <string.h>
#include "oidn_webgpu.h"
#include "oidn_webgpu_metal.h"
#include "oidn_webgpu_dx12.h"
_Static_assert(sizeof(oidnw_gpu_dx12_device_desc) == 48, "DX12 device ABI");
#if defined(OIDNW_TEST_VULKAN)
#include "oidn_webgpu_vulkan.h"
_Static_assert(sizeof(oidnw_gpu_vulkan_device_desc) == 112, "Vulkan device ABI");
#endif
_Static_assert(sizeof(oidnw_filter_desc) == 40, "CPU filter ABI 3.0.0");
_Static_assert(sizeof(oidnw_gpu_metal_device_desc) == 64, "Metal device ABI");
_Static_assert(sizeof(oidnw_gpu_submit_desc) == 112, "submit ABI");

_Static_assert(sizeof(oidnw_gpu_filter_desc) == 40, "GPU filter ABI 3.0.0");
_Static_assert(sizeof(oidnw_gpu_limits) == 40, "limits ABI 3.0.0");
_Static_assert(offsetof(oidnw_gpu_limits, next) == 0, "limits extension prefix");
_Static_assert(offsetof(oidnw_gpu_limits, image_offset_alignment) == 8, "limits alignment offset");
_Static_assert(offsetof(oidnw_gpu_limits, max_buffer_size) == 16, "limits buffer size offset");
_Static_assert(offsetof(oidnw_gpu_limits, max_image_bytes) == 24, "limits image bytes offset");
_Static_assert(offsetof(oidnw_gpu_limits, max_pending_submissions) == 32, "limits pending offset");
_Static_assert(sizeof(oidnw_gpu_prepared_info) == 48, "prepared info ABI 3.0.0");
_Static_assert(offsetof(oidnw_filter_desc, hdr) == 8, "CPU descriptor extension prefix");
_Static_assert(offsetof(oidnw_gpu_filter_desc, kernel) == 20, "GPU kernel offset");
_Static_assert(offsetof(oidnw_gpu_submit_desc, images) == 16, "submit images offset");
_Static_assert(offsetof(oidnw_gpu_prepared_info, workspace_bytes) == 24, "workspace bytes offset");

int main(void) {
  if (!OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version())) return 1;
  if (OIDNW_API_VERSION != 0x00030000u || OIDNW_API_VERSION_MAJOR != 3 ||
      OIDNW_API_VERSION_MINOR != 0 || OIDNW_API_VERSION_PATCH != 0) return 1;
  if (!OIDNW_API_VERSION_IS_COMPATIBLE(0x00030100u) ||
      !OIDNW_API_VERSION_IS_COMPATIBLE(0x000300ffu) ||
      OIDNW_API_VERSION_IS_COMPATIBLE(0x000200ffu) ||
      OIDNW_API_VERSION_IS_COMPATIBLE(0x00040000u)) return 1;
  const struct { uint32_t library, header; int compatible; } versions[] = {
    {0x00030000u, 0x00030000u, 1},
    {0x00030200u, 0x000301ffu, 1},
    {0x000300ffu, 0x00030100u, 0},
    {0x00030100u, 0x000301ffu, 1},
    {0x000301ffu, 0x00030100u, 1},
    {0x0002ffffu, 0x00030000u, 0},
    {0x00040000u, 0x00030100u, 0}
  };
  for (size_t i = 0; i < sizeof versions / sizeof versions[0]; ++i)
    if (oidnw_api_version_is_compatible(versions[i].library, versions[i].header) != versions[i].compatible) return 1;
  oidnw_gpu_dx12_device_desc dx12;
  oidnw_gpu_dx12_device_desc_default(&dx12);
  if (dx12.next || dx12.flags || dx12.host_device || dx12.agility_sdk_version || dx12.agility_sdk_path || dx12.max_pending_submissions != 64) return 1;
  oidnw_gpu_device *dx12_device = NULL;
  if (oidnw_gpu_device_create_dx12(&dx12, &dx12_device, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || dx12_device) return 1;
#if defined(OIDNW_TEST_VULKAN)
  _Static_assert(sizeof(oidnw_gpu_vulkan_requirements_desc) == 48, "Vulkan requirements ABI");
  oidnw_gpu_vulkan_requirements_desc query;
  oidnw_gpu_vulkan_requirements_desc_default(&query);
  if (query.next || query.flags || query.instance || query.physical_device || query.get_instance_proc_addr || query.instance_create_info) return 1;
  oidnw_gpu_vulkan_requirements *requirements = NULL;
  if (oidnw_gpu_vulkan_requirements_create(&query, &requirements, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || requirements) return 1;
  size_t extension_count = 99;
  if (oidnw_gpu_vulkan_instance_extensions(NULL, VK_API_VERSION_1_2, NULL, &extension_count, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || extension_count) return 1;
  oidnw_gpu_vulkan_requirements_destroy(NULL);
  oidnw_gpu_vulkan_device_desc vulkan;
  oidnw_gpu_vulkan_device_desc_default(&vulkan);
  if (vulkan.next || vulkan.instance || vulkan.device || vulkan.get_instance_proc_addr ||
      vulkan.lock || vulkan.unlock || vulkan.max_pending_submissions != 64) return 1;
  oidnw_gpu_device *gpu_device = NULL;
  if (oidnw_gpu_device_import_vulkan(&vulkan, &gpu_device, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || gpu_device) return 1;
#endif
  oidnw_gpu_metal_device_desc metal;
  oidnw_gpu_metal_device_desc_default(&metal);
  if (metal.next || metal.device || metal.queue || metal.lock || metal.unlock ||
      metal.max_pending_submissions != 64) return 1;
  oidnw_gpu_filter_desc gpu_filter;
  oidnw_gpu_filter_desc_default(&gpu_filter);
  if (gpu_filter.next || gpu_filter.kernel != OIDNW_KERNEL_AUTO || gpu_filter.quality != OIDNW_QUALITY_HIGH || gpu_filter.hdr != 1 || gpu_filter.flags != OIDNW_FILTER_RUNTIME_CHECKS ||
      gpu_filter.binding_cache_capacity != 8) return 1;
  oidnw_gpu_submit_desc submit;
  memset(&submit, 0xff, sizeof submit);
  oidnw_gpu_submit_desc_default(&submit);
  if (submit.next || submit.input_scale != 1 || submit.wait_count || submit.signal_count ||
      submit.images.color.buffer || submit.waits || submit.signals) return 1;
  uint64_t sequence = 9;
  if (oidnw_gpu_submit(NULL, &submit, &sequence, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || sequence != 0) return 1;
  uint32_t completed = 9;
  if (oidnw_gpu_wait(NULL, 1, 0, &completed, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || completed != 0) return 1;
  if (oidnw_gpu_device_destroy(NULL, NULL, 0) != OIDNW_OK) return 1;
  oidnw_gpu_limits limits = {0};
  if (oidnw_gpu_get_limits(NULL, &limits, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  if (oidnw_gpu_filter_warm_up(NULL, 16, 16, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  if (oidnw_filter_prepare(NULL, 16, 16, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  oidnw_filter_desc desc;
  memset(&desc, 0xff, sizeof desc);
  oidnw_filter_desc_default(&desc);
  if (desc.next || desc.hdr != 1 || desc.input_scale != 1.0f ||
      desc.transfer != OIDNW_TRANSFER_PU || desc.kernel != OIDNW_KERNEL_AUTO || desc.flags != 0) {
    fprintf(stderr, "header and shared-library defaults disagree\n");
    return 1;
  }
  oidnw_device *device = NULL;
  char error[256] = {0};
  int32_t status = oidnw_device_create(0xffffffffu, &device, error, sizeof error);
  if (status != OIDNW_ERROR_INVALID_ARGUMENT || device != NULL || error[0] == '\0') {
    fprintf(stderr, "shared-library error reporting failed\n");
    return 1;
  }
  const void *unknown = (const void *)(uintptr_t)1;
  dx12.next = unknown;
  metal.next = unknown;
  gpu_filter.next = unknown;
  submit.next = unknown;
  desc.next = unknown;
  oidnw_gpu_filter *gpu_handle = NULL;
  oidnw_filter *cpu_handle = NULL;
  if (oidnw_gpu_device_create_dx12(&dx12, &dx12_device, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
  if (oidnw_gpu_device_import_metal(&metal, &dx12_device, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
  if (oidnw_gpu_filter_create(NULL, NULL, &gpu_filter, &gpu_handle, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
  if (oidnw_gpu_submit(NULL, &submit, &sequence, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || sequence || !strstr(error, "extension")) return 1;
  if (oidnw_filter_create(NULL, "x", 1, &desc, &cpu_handle, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
#if defined(OIDNW_TEST_VULKAN)
  query.next = unknown;
  vulkan.next = unknown;
  if (oidnw_gpu_vulkan_requirements_create(&query, &requirements, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
  if (oidnw_gpu_device_import_vulkan(&vulkan, &dx12_device, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
#endif
  limits.next = unknown;
  if (oidnw_gpu_get_limits(NULL, &limits, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "limits extension")) return 1;
  limits.next = NULL;
  limits.image_offset_alignment = limits.max_pending_submissions = 99;
  limits.max_buffer_size = limits.max_image_bytes = 99;
  if (oidnw_gpu_get_limits(NULL, &limits, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT ||
      limits.next || limits.image_offset_alignment || limits.max_buffer_size ||
      limits.max_image_bytes || limits.max_pending_submissions) return 1;
  desc.next = NULL;
  const uint32_t invalid_flags[] = {2u, OIDNW_FILTER_RUNTIME_CHECKS | 2u, 0x80000000u, UINT32_MAX};
  for (size_t i = 0; i < sizeof invalid_flags / sizeof invalid_flags[0]; ++i) {
    desc.flags = invalid_flags[i];
    if (oidnw_filter_create(NULL, "x", 1, &desc, &cpu_handle, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "desc.flags")) return 1;
  }
  desc.flags = 0;
  desc.input_scale = 0;
  if (oidnw_filter_create(NULL, "x", 1, &desc, &cpu_handle, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "positive")) return 1;
  oidnw_gpu_prepared_info info = {0};
  info.next = unknown;
  if (oidnw_gpu_prepared_get_info(NULL, &info, error, sizeof error) != OIDNW_ERROR_INVALID_ARGUMENT || !strstr(error, "extension")) return 1;
  info.next = NULL;
  info.tile_count = 99;
  if (oidnw_gpu_prepared_get_info(NULL, &info, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || info.tile_count) return 1;
  if (oidnw_filter_set_input_scale(NULL, 1, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  oidnw_filter_destroy(NULL);
  oidnw_device_destroy(NULL);
  puts("C shared-library ABI: OK (no GPU required)");
  return 0;
}
