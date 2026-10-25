use oidn_webgpu::{DenoiseBuffers, ImageBuffer, wgpu};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const OFFSET: u64 = 256;
pub const SCALES: [f32; 3] = [1.0, 2.0, 0.5];
pub fn allocator_sample(device: &wgpu::Device, frame: usize) -> serde_json::Value {
    device.generate_allocator_report().map_or(serde_json::Value::Null,|r|serde_json::json!({
        "frame":frame,"live_allocations":r.allocations.len(),"allocated_bytes":r.total_allocated_bytes,"reserved_bytes":r.total_reserved_bytes
    }))
}
pub struct Options {
    pub frames: usize,
    pub size: u32,
    pub tile: u32,
    pub fp32: bool,
    pub recommended: bool,
    pub shared: bool,
    pub debug: bool,
    pub trace: bool,
    pub unchecked: bool,
    pub weights: String,
    pub adapter: String,
}
impl Options {
    pub fn parse() -> Result<Self> {
        let mut result = Self {
            frames: 1000,
            size: 16,
            tile: 0,
            fp32: false,
            recommended: true,
            shared: false,
            debug: true,
            trace: false,
            unchecked: false,
            weights: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../weights/rt_hdr_alb_nrm_small.tza"
            )
            .into(),
            adapter: "RTX 4060".into(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--frames" => result.frames = args.next().ok_or("missing frames")?.parse()?,
                "--size" => result.size = args.next().ok_or("missing size")?.parse()?,
                "--tile" => result.tile = args.next().ok_or("missing tile")?.parse()?,
                "--weights" => result.weights = args.next().ok_or("missing weights")?,
                "--adapter" => result.adapter = args.next().ok_or("missing adapter")?,
                "--fp32" => result.fp32 = true,
                "--no-recommended" => result.recommended = false,
                "--shared-queue" => result.shared = true,
                "--no-debug" => result.debug = false,
                "--trace" => result.trace = true,
                "--unchecked" => result.unchecked = true,
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        if !(1..=100_000).contains(&result.frames)
            || !(1..=1024).contains(&result.size)
            || (result.tile != 0 && !(16..=16384).contains(&result.tile))
            || result.frames as u64 * result.size as u64 * result.size as u64 * 16
                > 256 * 1024 * 1024
        {
            return Err("invalid frames/size/tile or validation flags exceed 256 MiB".into());
        }
        Ok(result)
    }
}
pub fn bindings<'a>(
    buffers: &'a [wgpu::Buffer],
    output: Option<&'a wgpu::Buffer>,
) -> DenoiseBuffers<'a> {
    let view = |buffer| ImageBuffer {
        buffer,
        offset: OFFSET,
    };
    DenoiseBuffers {
        color: view(&buffers[0]),
        albedo: Some(view(&buffers[1])),
        normal: Some(view(&buffers[2])),
        output: view(output.unwrap_or(&buffers[3])),
    }
}
// Native host shaders are compiled to SPIR-V independently of wgpu. The consumer
// checks every channel's bits; only these integer flags are CPU mapped.
pub const HOST_WGSL: &str = r#"
struct Params { size: u32, pattern: u32, frame: u32, unused: u32 }
var<immediate> p: Params;
@group(0) @binding(0) var<storage, read_write> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;
@compute @workgroup_size(64) fn produce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= p.size * p.size { return; }
    let noise = f32(((i + p.pattern * 31u) * 1664525u + 1013904223u) >> 24u) / 255.0 - 0.5;
    let signal = 0.5 + f32(i % p.size) / f32(p.size) + f32(p.pattern) * 0.25;
    a[i] = vec4<f32>(max(vec3<f32>(0.8, 0.6, 0.4) * signal + noise * 0.3, vec3<f32>(0)), 0.25 + f32(p.pattern) * 0.125);
    b[i] = vec4<f32>(0.8, 0.6, 0.4, 0);
    c[i] = vec4<f32>(0, 0, 1, 0);
}
@compute @workgroup_size(64) fn consume(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; let count = p.size * p.size; if i >= count { return; }
    let actual = a[i]; let expected = b[i];
    var flag = select(0u, 1u, any(bitcast<vec4<u32>>(actual) != bitcast<vec4<u32>>(expected)));
    if any(actual.xyz < vec3<f32>(0)) || any((bitcast<vec4<u32>>(actual) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u)) { flag |= 2u; }
    if actual.w != 0.25 + f32(p.pattern) * 0.125 { flag |= 4u; }
    if !any(actual.xyz > vec3<f32>(0)) { flag |= 8u; }
    // c is viewed as vec4<f32> so that all bindings have the same native layout.
    c[p.frame * count + i] = bitcast<vec4<f32>>(vec4<u32>(flag));
}
"#;

pub fn spirv(entry: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(HOST_WGSL)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    Ok(naga::back::spv::write_vec(
        &module,
        &info,
        &Default::default(),
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.into(),
        }),
    )?)
}
