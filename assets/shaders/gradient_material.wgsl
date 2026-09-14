#import bevy_pbr::forward_io::VertexOutput

struct GradientUniforms {
    s_min: f32,
    s_max: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(2) @binding(0)
var<uniform> gradient: GradientUniforms;

fn colormap(t: f32) -> vec3<f32> {
    // purple → magenta → orange (surface-gradient style)
    let x = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.25, 0.05, 0.45);
    let c1 = vec3<f32>(0.85, 0.15, 0.55);
    let c2 = vec3<f32>(1.00, 0.55, 0.10);
    if (x < 0.5) {
        return mix(c0, c1, x * 2.0);
    }
    return mix(c1, c2, (x - 0.5) * 2.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let span = max(gradient.s_max - gradient.s_min, 1.0e-20);
    let t = (in.color.r - gradient.s_min) / span;
    let rgb = colormap(t);
    return vec4<f32>(rgb, 1.0);
}
