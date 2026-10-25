/* Host Metal -> GPU C API -> host Metal, without CPU image transfers.
 * Build with the package's CMake examples, or from the repository root:
 * clang -fobjc-arc -I .native-build/package/include native/examples/metal_gpu.m \
 *   -L .native-build/package/lib -loidn_webgpu -framework Metal -framework Foundation \
 *   -Wl,-rpath,"$PWD/.native-build/package/lib" -o .native-build/metal_gpu
 * .native-build/metal_gpu weights/rt_hdr_alb_nrm_small.tza
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oidn_webgpu_metal.h"

static char error_text[1024];
static void check(int32_t status) {
  if (status != OIDNW_OK) {
    fprintf(stderr, "oidn GPU error %d: %s\n", status, error_text);
    exit(1);
  }
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
  NSError *error = nil;
  id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
  if (!result) { fprintf(stderr, "%s\n", error.description.UTF8String); exit(1); }
  return result;
}
int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc != 2) { fprintf(stderr, "usage: metal_gpu MODEL.tza\n"); return 2; }
    if (oidnw_api_version() != OIDNW_API_VERSION) {
      fprintf(stderr, "header/library version mismatch\n"); return 1;
    }
    const NSUInteger edge = 31, count = edge * edge, offset = 256, bytes = offset + count * 16;
    id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
    if (!metal) { fprintf(stderr, "Metal GPU required\n"); return 1; }
    id<MTLCommandQueue> producer = [metal newCommandQueueWithMaxCommandBufferCount:4096];
    id<MTLCommandQueue> consumer = [metal newCommandQueueWithMaxCommandBufferCount:4096];
    id<MTLSharedEvent> ready = [metal newSharedEvent], done = [metal newSharedEvent];
    id<MTLBuffer> buffers[4];
    for (NSUInteger i = 0; i < 4; ++i) buffers[i] = [metal newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> flags = [metal newBufferWithLength:count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!producer || !consumer || !ready || !done || !flags || !buffers[0] || !buffers[1] || !buffers[2] || !buffers[3]) {
      fprintf(stderr, "Metal allocation failed\n"); return 1;
    }
    memset(flags.contents, 0xff, flags.length); /* Validation flags only. */
    NSError *compile_error = nil;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void produce(device float4* color [[buffer(0)]], device float4* alb [[buffer(1)]], device float4* nrm [[buffer(2)]], uint i [[thread_position_in_grid]]) {"
      "if (i >= 961) return; color[i]=float4(0.8f+float(i%31)/31.0f,0.6f,0.4f,0.75f); alb[i]=float4(0.8f,0.6f,0.4f,0); nrm[i]=float4(0,0,1,0); }\n"
      "kernel void consume(device const float4* image [[buffer(0)]], device uint* flags [[buffer(1)]], uint i [[thread_position_in_grid]]) {"
      "if (i >= 961) return; float4 p=image[i]; flags[i]=(!all(isfinite(p)) || any(p.xyz<0.0f) || p.w!=0.75f || !any(p.xyz>0.0f)) ? 1u:0u; }";
    id<MTLLibrary> library = [metal newLibraryWithSource:source options:nil error:&compile_error];
    if (!library) { fprintf(stderr, "%s\n", compile_error.description.UTF8String); return 1; }
    id<MTLComputePipelineState> produce = pipeline(metal, library, @"produce");
    id<MTLComputePipelineState> consume = pipeline(metal, library, @"consume");

    oidnw_gpu_metal_device_desc device_desc;
    oidnw_gpu_metal_device_desc_default(&device_desc);
    device_desc.device = (__bridge void *)metal; /* Library creates a dedicated queue. */
    oidnw_gpu_device *device = NULL;
    check(oidnw_gpu_device_import_metal(&device_desc, &device, error_text, sizeof error_text));
    oidnw_gpu_limits limits;
    check(oidnw_gpu_get_limits(device, &limits, error_text, sizeof error_text));
    if (!limits.image_offset_alignment || offset % limits.image_offset_alignment ||
        bytes > limits.max_buffer_size || count * 16 > limits.max_image_bytes) {
      fprintf(stderr, "image exceeds imported device limits\n"); return 1;
    }
    oidnw_gpu_buffer *imported[4] = {0};
    for (NSUInteger i = 0; i < 4; ++i)
      check(oidnw_gpu_buffer_import_metal(device, (__bridge void *)buffers[i], bytes, &imported[i], error_text, sizeof error_text));
    oidnw_gpu_sync *ready_sync = NULL, *done_sync = NULL;
    check(oidnw_gpu_sync_import_metal(device, (__bridge void *)ready, &ready_sync, error_text, sizeof error_text));
    check(oidnw_gpu_sync_import_metal(device, (__bridge void *)done, &done_sync, error_text, sizeof error_text));
    NSData *weights = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
    if (!weights) { fprintf(stderr, "cannot read weights\n"); return 1; }
    oidnw_gpu_model *model = NULL;
    check(oidnw_gpu_model_create(weights.bytes, weights.length, &model, error_text, sizeof error_text));
    if (oidnw_gpu_model_input_count(model) != 3) { fprintf(stderr, "three-input model required\n"); return 1; }
    oidnw_gpu_filter_desc filter_desc;
    oidnw_gpu_filter_desc_default(&filter_desc);
    filter_desc.max_tile_size = 16;
    oidnw_gpu_filter *filter = NULL;
    check(oidnw_gpu_filter_create(device, model, &filter_desc, &filter, error_text, sizeof error_text));
    oidnw_gpu_model_destroy(model);
    oidnw_gpu_images images = {{imported[0], offset}, {imported[1], offset}, {imported[2], offset}, {imported[3], offset}};
    oidnw_gpu_prepared *prepared = NULL;
    check(oidnw_gpu_prepare(filter, (uint32_t)edge, (uint32_t)edge, &images, &prepared, error_text, sizeof error_text));
    oidnw_gpu_filter_destroy(filter);

    id<MTLCommandBuffer> production = [producer commandBuffer];
    id<MTLComputeCommandEncoder> encode = [production computeCommandEncoder];
    [encode setComputePipelineState:produce];
    for (NSUInteger i = 0; i < 3; ++i) [encode setBuffer:buffers[i] offset:offset atIndex:i];
    [encode dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encode endEncoding];
    [production encodeSignalEvent:ready value:1];
    [production commit];

    oidnw_gpu_timeline_point wait = {ready_sync, 1}, signal = {done_sync, 1};
    oidnw_gpu_submit_desc submit;
    oidnw_gpu_submit_desc_default(&submit);
    submit.images = images;
    submit.waits = &wait; submit.wait_count = 1;
    submit.signals = &signal; submit.signal_count = 1;
    uint64_t sequence = 0;
    check(oidnw_gpu_submit(prepared, &submit, &sequence, error_text, sizeof error_text));
    /* Releasing wrappers does not invalidate accepted GPU work. */
    oidnw_gpu_prepared_destroy(prepared);
    for (NSUInteger i = 0; i < 4; ++i) oidnw_gpu_buffer_destroy(imported[i]);
    oidnw_gpu_sync_destroy(ready_sync); oidnw_gpu_sync_destroy(done_sync);

    id<MTLCommandBuffer> consumption = [consumer commandBuffer];
    [consumption encodeWaitForEvent:done value:1];
    encode = [consumption computeCommandEncoder];
    [encode setComputePipelineState:consume];
    [encode setBuffer:buffers[3] offset:offset atIndex:0];
    [encode setBuffer:flags offset:0 atIndex:1];
    [encode dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encode endEncoding];
    [consumption commit];
    uint32_t complete = 0;
    check(oidnw_gpu_wait(device, sequence, UINT64_C(60000000000), &complete, error_text, sizeof error_text));
    [consumption waitUntilCompleted]; /* Test readback only, outside library execution. */
    if (consumption.status != MTLCommandBufferStatusCompleted || production.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "host Metal execution failed: %s\n", consumption.error.description.UTF8String); return 1;
    }
    const uint32_t *results = flags.contents;
    for (NSUInteger i = 0; i < count; ++i) if (results[i]) {
      fprintf(stderr, "invalid output at pixel %lu\n", (unsigned long)i); return 1;
    }
    check(oidnw_gpu_device_destroy(device, error_text, sizeof error_text));
    printf("{\"api\":\"Metal C\",\"sequence\":%" PRIu64 ",\"width\":31,\"height\":31,\"tile\":16,\"valid_output\":true}\n", sequence);
  }
  return 0;
}
