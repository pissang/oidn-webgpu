/* A GPU-independent shared-library/header ABI smoke test for CI packages. */
#include <stdio.h>
#include <string.h>
#include "oidn_webgpu.h"

int main(void) {
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
