struct Uniforms {
    vp: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    ambient: vec4<f32>,
    light_vp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
struct Input {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base: vec3<f32>,
    @location(3) props: vec3<f32>,
    @location(4) local: vec2<f32>,
    @location(5) emission: vec3<f32>,
};
struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base: vec3<f32>,
    @location(3) props: vec3<f32>,
    @location(4) local: vec2<f32>,
    @location(5) emission: vec3<f32>,
};
@vertex fn vs(v: Input) -> Output {
    var o: Output;
    o.position = u.vp * vec4(v.position,1.0);
    o.world=v.position; o.normal=v.normal; o.base=v.base; o.props=v.props; o.local=v.local; o.emission=v.emission;
    return o;
}
@vertex fn vs_shadow(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.light_vp * vec4(position, 1.0);
}
fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
fn noise(p: vec2<f32>) -> f32 {
    let cell=floor(p); let f=fract(p); let s=f*f*(3.0-2.0*f);
    return mix(mix(hash(cell),hash(cell+vec2(1.0,0.0)),s.x),mix(hash(cell+vec2(0.0,1.0)),hash(cell+vec2(1.0)),s.x),s.y);
}
/// Soft key-light visibility: 5x5 PCF over the fitted shadow map.
fn visibility(world: vec3<f32>, n: vec3<f32>, light: vec3<f32>) -> f32 {
    // Normal offset keeps lit faces from self-shadowing at grazing angles.
    // Normal offset scaled by texel footprint and grazing angle: without it, plates
    // nearly parallel to the light self-shadow in blotchy bands.
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    let offset = mix(0.05, 0.20, 1.0 - clamp(abs(dot(n, light)), 0.0, 1.0)) + texel * 4.0;
    let p = u.light_vp * vec4(world + n * offset, 1.0);
    let ndc = p.xyz / p.w;
    let uv = ndc.xy * vec2(0.5, -0.5) + vec2(0.5);
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) || ndc.z > 1.0 { return 1.0; }
    let step = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2(f32(x), f32(y)) * step * 1.5, ndc.z - 0.0035);
        }
    }
    return sum / 25.0;
}
/// A studio softbox above the camera plus a dim horizon: what polished metal reflects.
fn studio(r: vec3<f32>) -> f32 {
    let softbox = normalize(vec3(0.0, 0.45, 0.9));
    let band = exp(-pow((1.0 - dot(r, softbox)) / 0.012, 2.0));
    let sky = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);
    return 0.10 + 0.22 * sky * sky + 1.1 * band;
}
@fragment fn fs(v: Output) -> @location(0) vec4<f32> {
    var n=normalize(v.normal);
    let view=normalize(u.eye.xyz-v.world);
    let light=normalize(u.light.xyz);
    let half_vector=normalize(view+light);
    let rough=clamp(v.props.x,0.08,1.0); let metal=v.props.y;
    let brushed = v.props.z>0.5 && v.props.z<1.5;
    var grain=1.0;
    var streak=0.0;
    if brushed {
        // Horizontal machining at a scale the eye can resolve: broad tonal streaks,
        // finer lines faded out where they would alias below a pixel.
        let uv=select(v.world.xy,v.world.xz,abs(n.y)>0.7);
        // Fine, directional machining: no low-frequency blotches, only lines.
        let mid=noise(vec2(uv.x*0.55, uv.y*22.0));
        let fine=noise(vec2(uv.x*1.6, uv.y*54.0));
        let fine_aa=clamp(1.0-fwidth(uv.y*54.0)*0.6,0.0,1.0);
        streak=(mid-0.5)*0.22+(fine-0.5)*0.26*fine_aa;
        grain=1.0+streak*0.30;
    }
    if v.props.z>1.5 && v.props.z<2.5 {
        let radius=length(v.local);
        grain=1.0+sin(radius*200.0)*0.06*clamp(1.0-fwidth(radius)*120.0,0.0,1.0);
    }
    let nl=max(dot(n,light),0.0); let nv=max(dot(n,view),0.001); let nh=max(dot(n,half_vector),0.0);
    let a=rough*rough; let a2=a*a;
    let denom=nh*nh*(a2-1.0)+1.0;
    let distribution=a2/max(3.14159265*denom*denom,0.0001);
    let k=(rough+1.0)*(rough+1.0)/8.0;
    let geometry=(nl/(nl*(1.0-k)+k))*(nv/(nv*(1.0-k)+k));
    let f0=mix(vec3(0.04),v.base,metal);
    let fresnel=f0+(vec3(1.0)-f0)*pow(1.0-max(dot(half_vector,view),0.0),5.0);
    var spec=distribution*geometry*fresnel/max(4.0*nl*nv,0.001);
    if brushed {
        // Anisotropic lobe: brushing along X stretches the highlight across the grain.
        let t=normalize(vec3(1.0,0.0,0.0)-n*n.x);
        let th=dot(t,half_vector);
        let aniso=pow(max(sqrt(max(1.0-th*th,0.0)),0.0),90.0);
        spec+=f0*aniso*(0.55+streak*0.9);
    }
    if v.props.z>1.5 && v.props.z<2.5 {
        // Tangential machining produces a stable light-aligned anisotropic lobe.
        let tangent=normalize(vec3(-v.local.y,v.local.x,0.0001));
        let aniso=pow(max(1.0-abs(dot(tangent,half_vector)),0.0),12.0);
        spec+=f0*aniso*0.7;
    }
    let shadow=visibility(v.world,n,light);
    let direct=shadow*u.light.w;
    // Ambient falls off on downward faces and in the shadowed side of edges.
    let ambient=v.base*(0.22+0.55*max(n.y*0.5+0.5,0.0))*u.ambient.x*(0.75+0.25*shadow);
    var color=(ambient+v.base*grain*nl*(1.0-metal*0.5)*direct+spec*nl*direct)*1.25;
    if metal>0.2 {
        // Reflected studio light gives flat metal a gradient and a soft highlight band.
        let r=reflect(-view,n);
        let env=studio(r)*(1.0-rough*0.55)*(0.85+0.15*shadow);
        color+=f0*env*grain*0.9;
    }
    // Glass is deliberately dark and matte enough not to veil native labels.
    if v.props.z>2.5 { color=v.base*(0.75+0.25*nl)*(0.7+0.3*shadow); }
    if v.props.z>3.5 {
        let distance=length(v.local);
        let aa=max(fwidth(distance)*0.5,0.001);
        let dot=1.0-smoothstep(0.38-aa,0.38+aa,distance);
        let glow=(1.0-smoothstep(0.38,0.50,distance))*0.06;
        color=mix(color,v.emission,dot)+v.emission*glow;
    }
    return vec4(color,1.0);
}
