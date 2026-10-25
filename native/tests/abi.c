/* A GPU-independent shared-library/header ABI smoke test for CI packages. */
#include <stdio.h>
#include <string.h>
#include "oidn_webgpu.h"
#include "oidn_webgpu_metal.h"

int main(void) {
  if (oidnw_api_version() != OIDNW_API_VERSION) return 1;
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
  if (desc.size != sizeof desc || desc.hdr != 1 || desc.input_scale != 1.0f ||
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
