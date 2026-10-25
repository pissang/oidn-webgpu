// Wall time per image of the official Open Image Denoise library, with the
// protocol of vk_oidn's oidn_gpu_bench: device buffers, a warm-up batch, then
// a batch of asynchronous executions with a single synchronization.
//
// The library is loaded at run time, so only a C compiler is needed:
//   cc -O2 -o oidn_bench benchmarks/oidn-sdk/oidn_bench.c
//   ./oidn_bench <libOpenImageDenoise> <metal|cuda|hip|cpu> <width> <height> <base|large|fast>
//     [batch=50] [rounds=5] [warmup_batches=1] [weights.tza]
// batch=1 measures synchronized image latency. Inputs/output are GPU-resident
// RGBA f32 (RGB interpreted), HDR inputScale=1; preprocessing is included.
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#define LOAD(path) ((void*)LoadLibraryA(path))
#define SYMBOL(lib, name) ((void*)GetProcAddress((HMODULE)(lib), name))
#else
#include <dlfcn.h>
#define LOAD(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define SYMBOL(lib, name) dlsym(lib, name)
#endif

typedef void* Handle;
// Values from OpenImageDenoise/oidn.h (2.5.1).
enum { DEVICE_CPU = 1, DEVICE_CUDA = 3, DEVICE_HIP = 4, DEVICE_METAL = 5 };
enum { STORAGE_DEVICE = 2 };
enum { FORMAT_FLOAT3 = 3 };
enum { QUALITY_FAST = 4, QUALITY_HIGH = 6 };

static Handle (*oidnNewDevice)(int);
static void (*oidnCommitDevice)(Handle);
static int (*oidnGetDeviceError)(Handle, const char**);
static Handle (*oidnNewBufferWithStorage)(Handle, size_t, int);
static void (*oidnWriteBuffer)(Handle, size_t, size_t, const void*);
static void (*oidnReadBuffer)(Handle, size_t, size_t, void*);
static int (*oidnGetDeviceInt)(Handle, const char*);
static Handle (*oidnNewFilter)(Handle, const char*);
static void (*oidnSetFilterImage)(Handle, const char*, Handle, int, size_t, size_t, size_t, size_t, size_t);
static void (*oidnSetFilterBool)(Handle, const char*, bool);
static void (*oidnSetFilterInt)(Handle, const char*, int);
static void (*oidnSetFilterFloat)(Handle, const char*, float);
static void (*oidnSetSharedFilterData)(Handle, const char*, const void*, size_t);
static void (*oidnCommitFilter)(Handle);
static void (*oidnExecuteFilterAsync)(Handle);
static void (*oidnSyncDevice)(Handle);
static void (*oidnReleaseFilter)(Handle);
static void (*oidnReleaseBuffer)(Handle);
static void (*oidnReleaseDevice)(Handle);

static double now_ms(void) {
#ifdef _WIN32
  LARGE_INTEGER t, frequency;
  QueryPerformanceCounter(&t); QueryPerformanceFrequency(&frequency);
  return (double)t.QuadPart * 1e3 / frequency.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
#endif
}

static int ascending(const void* a, const void* b) {
  const double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void check(Handle device) {
  const char* message = NULL;
  if (oidnGetDeviceError(device, &message) != 0) {
    fprintf(stderr, "OIDN error: %s\n", message ? message : "unknown");
    exit(1);
  }
}

#define BIND(name)                                                   \
  do {                                                               \
    *(void**)&name = SYMBOL(lib, #name);                             \
    if (!name) { fprintf(stderr, "missing %s\n", #name); return 1; } \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <library> <metal|cuda|hip|cpu> <width> <height> <base|large|fast> [batch] [rounds] [warmup_batches] [weights.tza]\n", argv[0]);
    return 2;
  }
  void* lib = LOAD(argv[1]);
  if (!lib) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
  BIND(oidnNewDevice); BIND(oidnCommitDevice); BIND(oidnGetDeviceError);
  BIND(oidnNewBufferWithStorage); BIND(oidnWriteBuffer); BIND(oidnNewFilter);
  BIND(oidnReadBuffer); BIND(oidnGetDeviceInt);
  BIND(oidnSetFilterImage); BIND(oidnSetFilterBool); BIND(oidnSetFilterInt);
  BIND(oidnSetFilterFloat); BIND(oidnSetSharedFilterData);
  BIND(oidnCommitFilter); BIND(oidnExecuteFilterAsync); BIND(oidnSyncDevice);
  BIND(oidnReleaseFilter); BIND(oidnReleaseBuffer); BIND(oidnReleaseDevice);

  const char* kind = argv[2];
  int type = !strcmp(kind, "metal") ? DEVICE_METAL : !strcmp(kind, "cuda") ? DEVICE_CUDA
           : !strcmp(kind, "hip") ? DEVICE_HIP : !strcmp(kind, "cpu") ? DEVICE_CPU : 0;
  size_t width = (size_t)atoi(argv[3]), height = (size_t)atoi(argv[4]);
  const char* model = argv[5];
  int batch = argc > 6 ? atoi(argv[6]) : 50;
  int rounds = argc > 7 ? atoi(argv[7]) : 5;
  int warmup = argc > 8 ? atoi(argv[8]) : 1;
  if (!type || !width || !height || width > 16384 || height > 16384 || batch <= 0 || rounds <= 0 || rounds > 10000 || warmup < 0
      || (strcmp(model, "base") && strcmp(model, "large") && strcmp(model, "fast"))) return 2;

  Handle device = oidnNewDevice(type);
  check(device);
  oidnCommitDevice(device);
  check(device);

  // Same deterministic HDR gradient as compare.mjs and the native latency tool.
  size_t count = width * height * 4, bytes = count * sizeof(float);
  float* host = malloc(bytes);
  if (!host) return 1;
  Handle images[4];
  for (int i = 0; i < 4; i++) {
    images[i] = oidnNewBufferWithStorage(device, bytes, STORAGE_DEVICE);
    check(device);
    if (i < 3) {
      for (size_t y = 0; y < height; y++) for (size_t x = 0; x < width; x++) {
        float fx = (float)x / (width > 1 ? width - 1 : 1), fy = (float)y / (height > 1 ? height - 1 : 1);
        size_t j = (y * width + x) * 4;
        host[j] = i == 0 ? 0.1f + 4.f * fx : i == 1 ? 0.2f + 0.7f * fx : 0.f;
        host[j + 1] = i == 0 ? 0.05f + 2.f * fy : i == 1 ? 0.3f + 0.6f * fy : 0.f;
        host[j + 2] = i == 0 ? 0.2f + (float)((x * 17 + y * 13) % 97) / 97.f : i == 1 ? 0.55f : 1.f;
        host[j + 3] = 1.f;
      }
      oidnWriteBuffer(images[i], 0, bytes, host);
    }
  }

  Handle filter = oidnNewFilter(device, "RT");
  const char* names[4] = {"color", "albedo", "normal", "output"};
  for (int i = 0; i < 4; i++) oidnSetFilterImage(filter, names[i], images[i], FORMAT_FLOAT3, width, height, 0, 16, width * 16);
  oidnSetFilterBool(filter, "hdr", true);
  oidnSetFilterFloat(filter, "inputScale", 1.f);
  // The large network is selected by clean auxiliary images in high quality.
  oidnSetFilterBool(filter, "cleanAux", !strcmp(model, "large"));
  oidnSetFilterInt(filter, "quality", !strcmp(model, "fast") ? QUALITY_FAST : QUALITY_HIGH);
  void* weights = NULL;
  if (argc > 9) {
    FILE* file = fopen(argv[9], "rb");
    if (!file || fseek(file, 0, SEEK_END)) return 1;
    long size = ftell(file);
    if (size <= 0 || fseek(file, 0, SEEK_SET)) return 1;
    weights = malloc((size_t)size);
    if (!weights || fread(weights, 1, (size_t)size, file) != (size_t)size) return 1;
    fclose(file);
    oidnSetSharedFilterData(filter, "weights", weights, (size_t)size);
  }
  double started = now_ms();
  oidnCommitFilter(filter);
  check(device);
  double initialization = now_ms() - started;

  for (int w = 0; w < warmup; w++) {
    for (int i = 0; i < batch; i++) oidnExecuteFilterAsync(filter);
    oidnSyncDevice(device);
  }
  check(device);

  double best = 1e30, total = 0;
  double* samples = malloc((size_t)rounds * sizeof(double));
  if (!samples) return 1;
  for (int round = 0; round < rounds; round++) {
    double start = now_ms();
    for (int i = 0; i < batch; i++) oidnExecuteFilterAsync(filter);
    oidnSyncDevice(device);
    double perImage = (now_ms() - start) / batch;
    check(device);
    total += perImage;
    samples[round] = perImage;
    if (perImage < best) best = perImage;
  }
  oidnReadBuffer(images[3], 0, bytes, host);
  check(device);
  for (size_t i = 0; i < count; i++) if (i % 4 != 3 && !isfinite(host[i])) {
    fprintf(stderr, "non-finite OIDN output at %zu\n", i); return 1;
  }
  printf("{\"device\":\"%s\",\"version\":%d,\"width\":%zu,\"height\":%zu,\"model\":\"%s\",\"batch\":%d,\"bestMs\":%.6f,\"meanMs\":%.6f,\"initializationMs\":%.6f,\"timesMs\":[",
         kind, oidnGetDeviceInt(device, "version"), width, height, model, batch, best, total / rounds, initialization);
  check(device);
  for (int i = 0; i < rounds; i++) printf("%s%.6f", i ? "," : "", samples[i]);
  qsort(samples, rounds, sizeof(double), ascending);
  double median = rounds % 2 ? samples[rounds / 2] : (samples[rounds / 2 - 1] + samples[rounds / 2]) * 0.5;
  printf("],\"medianMs\":%.6f,\"p95Ms\":%.6f,\"finiteOutput\":true}\n", median, samples[(rounds * 95 + 99) / 100 - 1]);
  free(samples);
  oidnReleaseFilter(filter);
  for (int i = 0; i < 4; i++) oidnReleaseBuffer(images[i]);
  oidnReleaseDevice(device);
  free(weights);
  free(host);
  return 0;
}
