// GPU time per image of the official Open Image Denoise library, with the
// protocol of vk_oidn's oidn_gpu_bench: device buffers, a warm-up batch, then
// a batch of asynchronous executions with a single synchronization.
//
// The library is loaded at run time, so only a C compiler is needed:
//   cc -O2 -o oidn_bench benchmarks/oidn-sdk/oidn_bench.c
//   ./oidn_bench <libOpenImageDenoise> <device: metal|cuda|hip|cpu> <width> <height> <base|large|fast> [batch]
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
static Handle (*oidnNewFilter)(Handle, const char*);
static void (*oidnSetFilterImage)(Handle, const char*, Handle, int, size_t, size_t, size_t, size_t, size_t);
static void (*oidnSetFilterBool)(Handle, const char*, bool);
static void (*oidnSetFilterInt)(Handle, const char*, int);
static void (*oidnCommitFilter)(Handle);
static void (*oidnExecuteFilterAsync)(Handle);
static void (*oidnSyncDevice)(Handle);

static double now_ms(void) {
  struct timespec t;
  timespec_get(&t, TIME_UTC);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
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
    fprintf(stderr, "usage: %s <library> <metal|cuda|hip|cpu> <width> <height> <base|large|fast> [batch]\n", argv[0]);
    return 2;
  }
  void* lib = LOAD(argv[1]);
  if (!lib) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
  BIND(oidnNewDevice); BIND(oidnCommitDevice); BIND(oidnGetDeviceError);
  BIND(oidnNewBufferWithStorage); BIND(oidnWriteBuffer); BIND(oidnNewFilter);
  BIND(oidnSetFilterImage); BIND(oidnSetFilterBool); BIND(oidnSetFilterInt);
  BIND(oidnCommitFilter); BIND(oidnExecuteFilterAsync); BIND(oidnSyncDevice);

  const char* kind = argv[2];
  int type = !strcmp(kind, "metal") ? DEVICE_METAL : !strcmp(kind, "cuda") ? DEVICE_CUDA
           : !strcmp(kind, "hip") ? DEVICE_HIP : DEVICE_CPU;
  size_t width = (size_t)atoi(argv[3]), height = (size_t)atoi(argv[4]);
  const char* model = argv[5];
  int batch = argc > 6 ? atoi(argv[6]) : 50;

  Handle device = oidnNewDevice(type);
  check(device);
  oidnCommitDevice(device);
  check(device);

  // Inputs with the distributions of oidnBenchmark: color [0, 100], albedo [0, 1], normal [-1, 1].
  size_t count = width * height * 3, bytes = count * sizeof(float);
  float* host = malloc(bytes);
  Handle images[4];
  const float scale[3] = {100.f, 1.f, 2.f}, offset[3] = {0.f, 0.f, -1.f};
  srand(1);
  for (int i = 0; i < 4; i++) {
    images[i] = oidnNewBufferWithStorage(device, bytes, STORAGE_DEVICE);
    check(device);
    if (i < 3) {
      for (size_t j = 0; j < count; j++) host[j] = offset[i] + scale[i] * (float)rand() / (float)RAND_MAX;
      oidnWriteBuffer(images[i], 0, bytes, host);
    }
  }

  Handle filter = oidnNewFilter(device, "RT");
  const char* names[4] = {"color", "albedo", "normal", "output"};
  for (int i = 0; i < 4; i++) oidnSetFilterImage(filter, names[i], images[i], FORMAT_FLOAT3, width, height, 0, 0, 0);
  oidnSetFilterBool(filter, "hdr", true);
  // The large network is selected by clean auxiliary images in high quality.
  oidnSetFilterBool(filter, "cleanAux", !strcmp(model, "large"));
  oidnSetFilterInt(filter, "quality", !strcmp(model, "fast") ? QUALITY_FAST : QUALITY_HIGH);
  oidnCommitFilter(filter);
  check(device);

  for (int i = 0; i < batch; i++) oidnExecuteFilterAsync(filter);
  oidnSyncDevice(device);
  check(device);

  double best = 1e30, total = 0;
  const int rounds = 5;
  for (int round = 0; round < rounds; round++) {
    double start = now_ms();
    for (int i = 0; i < batch; i++) oidnExecuteFilterAsync(filter);
    oidnSyncDevice(device);
    double perImage = (now_ms() - start) / batch;
    check(device);
    total += perImage;
    if (perImage < best) best = perImage;
  }
  printf("{\"device\":\"%s\",\"width\":%zu,\"height\":%zu,\"model\":\"%s\",\"batch\":%d,\"bestMs\":%.3f,\"meanMs\":%.3f}\n",
         kind, width, height, model, batch, best, total / rounds);
  free(host);
  return 0;
}
