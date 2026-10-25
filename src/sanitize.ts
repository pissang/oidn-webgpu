/**
 * OIDN's input and output sanitizing: NaN becomes 0, then values are clamped
 * to their valid range. HDR values are bounded by FLT_MAX, so +inf becomes
 * FLT_MAX. Used before the network, by auto-exposure, and on the network
 * output before the inverse transfer.
 */
export const FLT_MAX = 3.4028234663852886e38;

export function sanitize(value: number, lo: number, hi: number) {
  return Number.isNaN(value) ? 0 : Math.min(Math.max(value, lo), hi);
}

/**
 * WGSL declarations of `FLT_MAX` and `fn sanitize(v: vec3f, lo: f32, hi: f32)`.
 * WGSL compilers may assume floats are finite, so NaN and infinity are
 * detected from the bits rather than with comparisons. FLT_MAX is spelled out
 * so the native core does not depend on scriptc's number formatting.
 */
export const sanitizeWGSL = /* wgsl */ `
const FLT_MAX = 3.4028234663852886e38;
fn sanitize(v: vec3f, lo: f32, hi: f32) -> vec3f {
  let bits = bitcast<vec3u>(v);
  let magnitude = bits & vec3u(0x7fffffffu);
  let negative = (bits & vec3u(0x80000000u)) != vec3u(0u);
  let infinite = select(vec3f(hi), vec3f(lo), negative);
  let clamped = select(clamp(v, vec3f(lo), vec3f(hi)), infinite, magnitude == vec3u(0x7f800000u));
  return select(clamped, vec3f(0.0), magnitude > vec3u(0x7f800000u));
}
`;
