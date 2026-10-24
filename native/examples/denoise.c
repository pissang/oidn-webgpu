/*
 * Denoises raw RGBA f32 images with the oidn-webgpu C API.
 *
 *   cc -O2 -I native/include native/examples/denoise.c \
 *     -L native/target/release -loidn_webgpu -Wl,-rpath,native/target/release -o denoise
 *   ./denoise weights.tza width height color.f32 albedo.f32 normal.f32 output.f32
 *     [quality 0-2] [max_tile_size] [runs] [kernel 0-3]
 *
 * albedo/normal may be "-" for models without them. Prints the time of the
 * first and following executions after explicit preparation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "oidn_webgpu.h"

static void *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file) return NULL;
  fseek(file, 0, SEEK_END);
  *size = (size_t)ftell(file);
  fseek(file, 0, SEEK_SET);
  void *data = malloc(*size);
  if (fread(data, 1, *size, file) != *size) {
    free(data);
    data = NULL;
  }
  fclose(file);
  return data;
}

static double now_ms(void) {
  struct timespec t;
  timespec_get(&t, TIME_UTC);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define CHECK(call)                                   \
  do {                                                \
    if ((call) != OIDNW_OK) {                         \
      fprintf(stderr, "%s failed: %s\n", #call, error); \
      return 1;                                       \
    }                                                 \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s weights.tza width height color albedo normal output [quality] [max_tile] [runs]\n", argv[0]);
    return 2;
  }
  char error[1024];
  size_t weights_size = 0;
  void *weights = read_file(argv[1], &weights_size);
  if (!weights) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  uint32_t width = (uint32_t)atoi(argv[2]), height = (uint32_t)atoi(argv[3]);
  size_t bytes = (size_t)width * height * 16;
  float *images[3] = {NULL, NULL, NULL};
  for (int i = 0; i < 3; i++) {
    if (strcmp(argv[4 + i], "-") == 0) continue;
    size_t size = 0;
    images[i] = read_file(argv[4 + i], &size);
    if (!images[i] || size != bytes) {
      fprintf(stderr, "%s must hold %zu bytes\n", argv[4 + i], bytes);
      return 1;
    }
  }
  float *output = malloc(bytes);

  oidnw_device *device = NULL;
  CHECK(oidnw_device_create(OIDNW_BACKEND_AUTO, &device, error, sizeof error));
  char name[256];
  uint32_t backend = 0, features = 0;
  oidnw_device_info(device, name, sizeof name, &backend, &features);
  printf("%s, backend %u, features %#x\n", name, backend, features);

  oidnw_filter_desc desc;
  oidnw_filter_desc_default(&desc);
  if (argc > 8) desc.quality = (uint32_t)atoi(argv[8]);
  if (argc > 9) desc.max_tile_size = (uint32_t)atoi(argv[9]);
  int runs = argc > 10 ? atoi(argv[10]) : 5;
  if (argc > 11) desc.kernel = (uint32_t)atoi(argv[11]);
  oidnw_filter *filter = NULL;
  CHECK(oidnw_filter_create(device, weights, weights_size, &desc, &filter, error, sizeof error));
  printf("model inputs: %u\n", oidnw_filter_input_count(filter));

  CHECK(oidnw_filter_prepare(filter, width, height, error, sizeof error));

  for (int run = 0; run <= runs; run++) {
    double start = now_ms();
    CHECK(oidnw_filter_execute(filter, width, height, images[0], images[1], images[2], output,
      error, sizeof error));
    printf("%s %.2f ms\n", run == 0 ? "first execute" : "execute", now_ms() - start);
  }
  FILE *file = fopen(argv[7], "wb");
  fwrite(output, 1, bytes, file);
  fclose(file);

  oidnw_filter_destroy(filter);
  oidnw_device_destroy(device);
  for (int i = 0; i < 3; i++) free(images[i]);
  free(output);
  free(weights);
  return 0;
}
