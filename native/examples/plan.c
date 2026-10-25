/*
 * Plans a denoising with the planner-only library and prints the program's
 * size and resource counts, for hosts that replay programs on their own GPU.
 *
 *   cc -O2 -I native/include native/examples/plan.c .native-build/liboidn_webgpu_core.a -o plan
 *   ./plan weights.tza width height
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oidn_webgpu_core.h"

static uint32_t word(const uint8_t *data, size_t index) {
  uint32_t value;
  memcpy(&value, data + index * 4, 4);
  return value;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s weights.tza width height\n", argv[0]);
    return 2;
  }
  FILE *file = fopen(argv[1], "rb");
  if (!file) return 1;
  fseek(file, 0, SEEK_END);
  size_t size = (size_t)ftell(file);
  fseek(file, 0, SEEK_SET);
  uint8_t *weights = malloc(size);
  if (fread(weights, 1, size, file) != size) return 1;
  fclose(file);

  /* A device with shader-f16 and WebGPU's default compute limits and no
   * subgroup matrices: the planner then uses the implicit GEMM kernels. */
  uint32_t config[] = {
    1, (uint32_t)atoi(argv[2]), (uint32_t)atoi(argv[3]), 0, 0,
    1, 16384, 256, 256, 256, 65535, 0, 0,
    0,          /* matrix configurations */
    0, 0, 0,    /* subgroup-matrix options */
    1, 0, 0, 0  /* hdr, PU transfer, whole image, input scale 1 */
  };
  oidnw_core_init();
  const uint8_t *program = NULL;
  size_t program_size = 0;
  oidnw_core_record_denoise(weights, size, (const uint8_t *)config, sizeof config, &program, &program_size);
  if (!program || program_size < 8 || word(program, 0) != 0x4e44494fu) {
    fprintf(stderr, "planning failed: %.*s\n", program_size > 8 ? (int)word(program, 1) : 0,
      program_size > 8 ? (const char *)program + 8 : "");
    return 1;
  }
  uint32_t inputs = word(program, 2);
  size_t index = 3 + inputs + 1 + 1 + 3;
  printf("program %zu bytes, %u inputs, %u buffers\n", program_size, inputs, word(program, index));
  oidnw_core_reset();
  free(weights);
  return 0;
}
