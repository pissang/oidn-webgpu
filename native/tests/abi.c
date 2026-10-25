/* A GPU-independent shared-library/header ABI smoke test for CI packages. */
#include <stdio.h>
#include <string.h>
#include "oidn_webgpu.h"
#include "oidn_webgpu_metal.h"
#include "oidn_webgpu_dx12.h"
_Static_assert(sizeof(oidnw_gpu_dx12_device_desc) == 40, "DX12 device ABI");
#if defined(OIDNW_TEST_VULKAN)
#include "oidn_webgpu_vulkan.h"
_Static_assert(sizeof(oidnw_gpu_vulkan_device_desc) == 104, "Vulkan device ABI");
#endif
_Static_assert(sizeof(oidnw_filter_desc) == 28, "CPU filter ABI 2.0.0");
_Static_assert(sizeof(oidnw_gpu_metal_device_desc) == 56, "Metal device ABI");
_Static_assert(sizeof(oidnw_gpu_submit_desc) == 104, "submit ABI");

int main(void) {
  if (oidnw_api_version() != OIDNW_API_VERSION) return 1;
  oidnw_gpu_dx12_device_desc dx12;
  oidnw_gpu_dx12_device_desc_default(&dx12);
  if (dx12.flags || dx12.host_device || dx12.agility_sdk_version || dx12.agility_sdk_path || dx12.max_pending_submissions != 64) return 1;
  oidnw_gpu_device *dx12_device = NULL;
  if (oidnw_gpu_device_create_dx12(&dx12, &dx12_device, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || dx12_device) return 1;
#if defined(OIDNW_TEST_VULKAN)
  _Static_assert(sizeof(oidnw_gpu_vulkan_requirements_desc) == 40, "Vulkan requirements ABI");
  oidnw_gpu_vulkan_requirements_desc query;
  oidnw_gpu_vulkan_requirements_desc_default(&query);
  if (query.flags || query.instance || query.physical_device || query.get_instance_proc_addr || query.instance_create_info) return 1;
  oidnw_gpu_vulkan_requirements *requirements = NULL;
  if (oidnw_gpu_vulkan_requirements_create(&query, &requirements, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || requirements) return 1;
  size_t extension_count = 99;
  if (oidnw_gpu_vulkan_instance_extensions(NULL, VK_API_VERSION_1_2, NULL, &extension_count, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || extension_count) return 1;
  oidnw_gpu_vulkan_requirements_destroy(NULL);
  oidnw_gpu_vulkan_device_desc vulkan;
  oidnw_gpu_vulkan_device_desc_default(&vulkan);
  if (vulkan.instance || vulkan.device || vulkan.get_instance_proc_addr ||
      vulkan.lock || vulkan.unlock || vulkan.max_pending_submissions != 64) return 1;
  oidnw_gpu_device *gpu_device = NULL;
  if (oidnw_gpu_device_import_vulkan(&vulkan, &gpu_device, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || gpu_device) return 1;
#endif
  oidnw_gpu_metal_device_desc metal;
  oidnw_gpu_metal_device_desc_default(&metal);
  if (metal.device || metal.queue || metal.lock || metal.unlock ||
      metal.max_pending_submissions != 64) return 1;
  oidnw_gpu_filter_desc gpu_filter;
  oidnw_gpu_filter_desc_default(&gpu_filter);
  if (gpu_filter.hdr != 1 || gpu_filter.flags != OIDNW_FILTER_RUNTIME_CHECKS ||
      gpu_filter.binding_cache_capacity != 8) return 1;
  oidnw_gpu_submit_desc submit;
  memset(&submit, 0xff, sizeof submit);
  oidnw_gpu_submit_desc_default(&submit);
  if (submit.input_scale != 1 || submit.wait_count || submit.signal_count ||
      submit.images.color.buffer || submit.waits || submit.signals) return 1;
  uint64_t sequence = 9;
  if (oidnw_gpu_submit(NULL, &submit, &sequence, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || sequence != 0) return 1;
  uint32_t completed = 9;
  if (oidnw_gpu_wait(NULL, 1, 0, &completed, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT || completed != 0) return 1;
  if (oidnw_gpu_device_destroy(NULL, NULL, 0) != OIDNW_OK) return 1;
  oidnw_gpu_limits limits;
  if (oidnw_gpu_get_limits(NULL, &limits, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  if (oidnw_filter_prepare(NULL, 16, 16, NULL, 0) != OIDNW_ERROR_INVALID_ARGUMENT) return 1;
  oidnw_filter_desc desc;
  memset(&desc, 0xff, sizeof desc);
  oidnw_filter_desc_default(&desc);
  if (desc.hdr != 1 || desc.input_scale != 1.0f ||
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
  oidnw_filter_destroy(NULL);
  oidnw_device_destroy(NULL);
  puts("C shared-library ABI: OK (no GPU required)");
  return 0;
}
