
RWStructuredBuffer<float4> a : register(u0);
RWStructuredBuffer<float4> b : register(u1);
RWStructuredBuffer<float4> c : register(u2);
cbuffer Params : register(b0) { uint size, pattern, frame, unused; }
[numthreads(64,1,1)] void produce(uint3 id : SV_DispatchThreadID) {
    uint i=id.x; if(i>=size*size) return;
    float noise=float(((i+pattern*31u)*1664525u+1013904223u)>>24u)/255.0f-0.5f;
    float signal=0.5f+float(i%size)/float(size)+float(pattern)*0.25f;
    a[i]=float4(max(float3(0.8f,0.6f,0.4f)*signal+noise*0.3f,0.0f),0.25f+float(pattern)*0.125f);
    b[i]=float4(0.8f,0.6f,0.4f,0.0f); c[i]=float4(0.0f,0.0f,1.0f,0.0f);
}
[numthreads(64,1,1)] void consume(uint3 id : SV_DispatchThreadID) {
    uint i=id.x, count=size*size; if(i>=count) return;
    float4 actual=a[i], expected=b[i]; uint flag=any(asuint(actual)!=asuint(expected))?1u:0u;
    if(!all(isfinite(actual)) || any(actual.xyz<0)) flag|=2u;
    if(actual.w!=0.25f+float(pattern)*0.125f) flag|=4u;
    if(!any(actual.xyz>0)) flag|=8u;
    c[frame*count+i]=asfloat(uint4(flag,flag,flag,flag));
}
[numthreads(64,1,1)] void initialize(uint3 id : SV_DispatchThreadID) {
    uint i=id.x+id.y*(65535u*64u); if(i>=size) return;
    c[i]=asfloat(uint4(0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu));
}
