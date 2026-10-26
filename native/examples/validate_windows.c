/* Independent MSVC host for a precompiled oidn-webgpu Windows DLL.
 * Link only the package header/import library; ship its DLL beside this exe.
 * args: weights width height backend kernel checks tile warmups runs output [input-dir]
 * Input-dir, when supplied, contains color.f32/albedo.f32/normal.f32 (RGBA32F).
 * Timings include uploads, GPU work and readback; validation is outside timing.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oidn_webgpu.h"

static char error[2048];
#define REQUIRE(condition, message) do { if (!(condition)) { \
  fprintf(stderr, "%s: %s\n", message, error); exit(1); } } while (0)
#define CALL(expression) REQUIRE((expression) == OIDNW_OK, #expression)

static double now_ms(void) {
  LARGE_INTEGER t, frequency;
  QueryPerformanceCounter(&t); QueryPerformanceFrequency(&frequency);
  return (double)t.QuadPart * 1000.0 / (double)frequency.QuadPart;
}

static void *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  REQUIRE(file != NULL, path);
  REQUIRE(fseek(file, 0, SEEK_END) == 0, "seek");
  long length = ftell(file);
  REQUIRE(length > 0 && fseek(file, 0, SEEK_SET) == 0, "file size");
  void *bytes = malloc((size_t)length);
  REQUIRE(bytes != NULL, "allocate file");
  REQUIRE(fread(bytes, 1, (size_t)length, file) == (size_t)length, "read file");
  fclose(file); *size = (size_t)length;
  return bytes;
}

static void validate(const float *output, const float *color, size_t pixels) {
  for (size_t pixel = 0; pixel < pixels; pixel++) {
    for (size_t c = 0; c < 3; c++) {
      float value = output[pixel * 4 + c];
      if (!isfinite(value) || value < 0) {
        fprintf(stderr, "invalid RGB at pixel %zu channel %zu: %g\n", pixel, c, value);
        exit(1);
      }
    }
    REQUIRE(output[pixel * 4 + 3] == color[pixel * 4 + 3], "alpha changed");
  }
}

int main(int argc, char **argv) {
  REQUIRE(argc >= 11, "weights width height backend kernel checks tile warmups runs output [input-dir]");
  uint32_t width = (uint32_t)atoi(argv[2]), height = (uint32_t)atoi(argv[3]);
  uint32_t backend = (uint32_t)atoi(argv[4]), kernel = (uint32_t)atoi(argv[5]);
  uint32_t checks = (uint32_t)atoi(argv[6]), tile = (uint32_t)atoi(argv[7]);
  int warmups = atoi(argv[8]), runs = atoi(argv[9]);
  REQUIRE(width && height && width <= 16384 && height <= 16384 && runs > 0 && runs <= 10000 && warmups >= 0,
    "invalid dimensions or runs");
  REQUIRE((backend == OIDNW_BACKEND_VULKAN || backend == OIDNW_BACKEND_DX12) &&
    kernel <= OIDNW_KERNEL_SUBGROUP_MATRIX && checks <= 1, "invalid backend, kernel or runtime checks");
  REQUIRE(OIDNW_API_VERSION_IS_COMPATIBLE(oidnw_api_version()), "API version mismatch");
  size_t pixels = (size_t)width * height, bytes = pixels * 4 * sizeof(float), weights_size;
  void *weights = read_file(argv[1], &weights_size);
  float *images[3];
  const char *names[] = {"color", "albedo", "normal"};
  for (int i = 0; i < 3; i++) {
    if (argc > 11) {
      char file[4096]; size_t length;
      int written = snprintf(file, sizeof file, "%s/%s.f32", argv[11], names[i]);
      REQUIRE(written > 0 && (size_t)written < sizeof file, "input path too long");
      images[i] = read_file(file, &length);
      REQUIRE(length == bytes, "input byte count mismatch");
    } else {
      images[i] = malloc(bytes); REQUIRE(images[i] != NULL, "allocate input");
      for (uint32_t y = 0; y < height; y++) for (uint32_t x = 0; x < width; x++) {
        float fx = (float)x / (width > 1 ? width - 1 : 1), fy = (float)y / (height > 1 ? height - 1 : 1);
        size_t p = ((size_t)y * width + x) * 4;
        images[i][p] = i == 0 ? 0.1f + 4.f * fx : i == 1 ? 0.2f + 0.7f * fx : 0.f;
        images[i][p + 1] = i == 0 ? 0.05f + 2.f * fy : i == 1 ? 0.3f + 0.6f * fy : 0.f;
        images[i][p + 2] = i == 0 ? 0.2f + (float)((x * 17 + y * 13) % 97) / 97.f : i == 1 ? 0.55f : 1.f;
        images[i][p + 3] = 1.f;
      }
    }
  }
  float *output = malloc(bytes), *first = malloc(bytes);
  double *times = malloc((size_t)runs * sizeof(double));
  REQUIRE(output && first && times, "allocate output");
  oidnw_device *device = NULL;
  double started = now_ms();
  CALL(oidnw_device_create(backend, &device, error, sizeof error));
  double device_ms = now_ms() - started;
  char adapter[256]; uint32_t actual_backend = 0, features = 0;
  CALL(oidnw_device_info(device, adapter, sizeof adapter, &actual_backend, &features));
  REQUIRE(actual_backend == backend, "unexpected backend");
  const char *expected = getenv("OIDNW_EXPECT_ADAPTER");
  if (expected && *expected) REQUIRE(strstr(adapter, expected) != NULL, "unexpected GPU adapter");
  if (backend == OIDNW_BACKEND_VULKAN && kernel == OIDNW_KERNEL_SUBGROUP_MATRIX)
    REQUIRE(features & OIDNW_FEATURE_COOPERATIVE_MATRIX, "missing cooperative matrix feature");
  oidnw_filter_desc desc;
  oidnw_filter_desc_default(&desc);
  desc.kernel = kernel; desc.max_tile_size = tile;
  desc.flags = checks ? OIDNW_FILTER_RUNTIME_CHECKS : 0;
  oidnw_filter *filter = NULL;
  CALL(oidnw_filter_create(device, weights, weights_size, &desc, &filter, error, sizeof error));
  REQUIRE(oidnw_filter_input_count(filter) == 3, "expected three input images");
  fprintf(stderr, "Preparing %s, backend %u, kernel %u, runtime checks %u, %ux%u, tile %u\n",
    adapter, actual_backend, kernel, checks, width, height, tile);
  started = now_ms();
  CALL(oidnw_filter_prepare(filter, width, height, error, sizeof error));
  double prepare_ms = now_ms() - started;
  fprintf(stderr, "Prepared in %.2f ms; executing\n", prepare_ms);
  started = now_ms();
  CALL(oidnw_filter_execute(filter, width, height, images[0], images[1], images[2], output, error, sizeof error));
  double first_ms = now_ms() - started;
  validate(output, images[0], pixels);
  memcpy(first, output, bytes);
  for (int run = -warmups; run < runs; run++) {
    memset(output, 0xff, bytes);
    started = now_ms();
    CALL(oidnw_filter_execute(filter, width, height, images[0], images[1], images[2], output, error, sizeof error));
    double elapsed = now_ms() - started;
    if (run >= 0) times[run] = elapsed;
    validate(output, images[0], pixels);
    REQUIRE(memcmp(first, output, bytes) == 0, "repeated RGBA output differs");
  }
  FILE *file = fopen(argv[10], "wb");
  REQUIRE(file != NULL, "create output");
  REQUIRE(fwrite(output, 1, bytes, file) == bytes, "write output");
  fclose(file);
  char dll[4096];
  DWORD dll_length = GetModuleFileNameA(GetModuleHandleA("oidn_webgpu.dll"), dll, sizeof dll);
  REQUIRE(dll_length > 0 && dll_length < sizeof dll, "loaded DLL path");
  for (char *p = dll; *p; p++) if (*p == '\\') *p = '/';
  printf("{\"adapter\":\"%s\",\"backend\":%u,\"features\":%u,\"kernel\":%u,\"checks\":%u,\"tile\":%u,"
    "\"width\":%u,\"height\":%u,\"dll\":\"%s\",\"deviceMs\":%.6f,\"prepareMs\":%.6f,\"firstMs\":%.6f,"
    "\"finiteNonnegative\":true,\"alphaPreserved\":true,\"repeatIdentical\":true,\"timesMs\":[",
    adapter, actual_backend, features, kernel, checks, tile, width, height, dll, device_ms, prepare_ms, first_ms);
  for (int run = 0; run < runs; run++) printf("%s%.6f", run ? "," : "", times[run]);
  puts("]}");
  oidnw_filter_destroy(filter); oidnw_device_destroy(device);
  for (int i = 0; i < 3; i++) free(images[i]);
  free(output); free(first); free(times); free(weights);
  return 0;
}
