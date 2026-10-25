/* Host Metal buffers with the common C++ GPU lifecycle. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oidn_webgpu_metal.hpp"

static oidnw::gpu::detached_device deferred_context;
static void check(const oidnw::gpu::status &status) {
  oidnw::gpu::require(status.code, status.message.data());
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
  NSError *error = nil;
  id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
  if (!result) throw std::runtime_error(error.description.UTF8String);
  return result;
}
int main(int argc, const char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2) { fprintf(stderr, "usage: metal_gpu_cpp MODEL.tza\n"); return 2; }
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
      memset(flags.contents, 0xff, flags.length);
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
      auto device = oidnw::metal::import_device(device_desc);
      auto limits = device.limits();
      if (!limits.image_offset_alignment || offset % limits.image_offset_alignment ||
          bytes > limits.max_buffer_size || count * 16 > limits.max_image_bytes)
        throw std::runtime_error("image exceeds device limits");
      std::array<oidnw::gpu::buffer, 4> imported;
      for (NSUInteger i = 0; i < 4; ++i)
        imported[i] = oidnw::metal::import_buffer(device, (__bridge void *)buffers[i], bytes);
      auto ready_sync = oidnw::metal::import_sync(device, (__bridge void *)ready);
      auto done_sync = oidnw::metal::import_sync(device, (__bridge void *)done);
      NSData *weights = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
      if (!weights) { fprintf(stderr, "cannot read weights\n"); return 1; }
      auto model = oidnw::gpu::load_model(weights.bytes, weights.length);
      if (oidnw_gpu_model_input_count(model.get()) != 3)
        throw std::runtime_error("three-input model required");
      auto options = oidnw::gpu::filter_options();
      options.max_tile_size = 16;
      auto filter = oidnw::gpu::make_filter(device, model, options);
      oidnw_gpu_images images = {{imported[0].get(), offset}, {imported[1].get(), offset},
          {imported[2].get(), offset}, {imported[3].get(), offset}};
      auto prepared = oidnw::gpu::prepare(filter, (uint32_t)edge, (uint32_t)edge, images);
      model.reset();
      filter.reset();

      id<MTLCommandBuffer> production = [producer commandBuffer];
      id<MTLComputeCommandEncoder> encode = [production computeCommandEncoder];
      [encode setComputePipelineState:produce];
      for (NSUInteger i = 0; i < 3; ++i) [encode setBuffer:buffers[i] offset:offset atIndex:i];
      [encode dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [encode endEncoding];
      [production encodeSignalEvent:ready value:1];
      [production commit];

      oidnw_gpu_timeline_point wait = {ready_sync.get(), 1}, signal = {done_sync.get(), 1};
      auto submitted = prepared.submit(images, wait, signal);
      check(submitted);
      prepared.reset();
      for (auto &buffer : imported) buffer.reset();
      ready_sync.reset();
      done_sync.reset();

      id<MTLCommandBuffer> consumption = [consumer commandBuffer];
      [consumption encodeWaitForEvent:done value:1];
      encode = [consumption computeCommandEncoder];
      [encode setComputePipelineState:consume];
      [encode setBuffer:buffers[3] offset:offset atIndex:0];
      [encode setBuffer:flags offset:0 atIndex:1];
      [encode dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [encode endEncoding];
      [consumption commit];
      auto completion = device.wait(submitted.sequence, UINT64_C(60000000000));
      if (!completion.completed) {
        deferred_context = device.release();
        fprintf(stderr, "context retained: %s\n", completion.message.data());
        return 1;
      }
      check(completion);
      [consumption waitUntilCompleted];
      if (consumption.status != MTLCommandBufferStatusCompleted || production.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "host Metal execution failed: %s\n", consumption.error.description.UTF8String); return 1;
      }
      const uint32_t *results = static_cast<const uint32_t *>(flags.contents);
      for (NSUInteger i = 0; i < count; ++i) if (results[i]) {
        fprintf(stderr, "invalid output at pixel %lu\n", (unsigned long)i); return 1;
      }
      check(device.close(0));
      printf("{\"api\":\"Metal C++\",\"sequence\":%" PRIu64 ",\"width\":31,\"height\":31,\"tile\":16,\"valid_output\":true}\n", submitted.sequence);
    } catch (const std::exception &error) {
      fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
  return 0;
}
