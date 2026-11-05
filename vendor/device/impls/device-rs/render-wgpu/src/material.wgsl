struct Uniforms {
    vp: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    ambient: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
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
fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
fn noise(p: vec2<f32>) -> f32 {
    let cell=floor(p); let f=fract(p); let s=f*f*(3.0-2.0*f);
    return mix(mix(hash(cell),hash(cell+vec2(1.0,0.0)),s.x),mix(hash(cell+vec2(0.0,1.0)),hash(cell+vec2(1.0)),s.x),s.y);
}
@fragment fn fs(v: Output) -> @location(0) vec4<f32> {
    var n=normalize(v.normal);
    let view=normalize(u.eye.xyz-v.world);
    let light=normalize(u.light.xyz);
    let half_vector=normalize(view+light);
    let rough=clamp(v.props.x,0.08,1.0); let metal=v.props.y;
    var grain=1.0;
    if v.props.z>0.5 && v.props.z<1.5 {
        // Horizontal machining, attenuated at subpixel pitch rather than crawling with resolution.
        let uv=select(v.world.xy,v.world.xz,abs(n.y)>0.7);
        let aa=clamp(1.0-length(fwidth(uv*vec2(0.65,42.0)))*0.25,0.0,1.0);
        let g=noise(uv*vec2(0.65,42.0));
        grain=1.0+(g-0.5)*0.65*aa;
        n=normalize(n+vec3(0.0,(g-0.5)*0.12*aa,0.0));
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
    if v.props.z>1.5 && v.props.z<2.5 {
        // Tangential machining produces a stable light-aligned anisotropic lobe.
        let tangent=normalize(vec3(-v.local.y,v.local.x,0.0001));
        let aniso=pow(max(1.0-abs(dot(tangent,half_vector)),0.0),12.0);
        spec+=f0*aniso*0.7;
    }
    let ambient=v.base*(0.30+0.65*max(n.y,0.0))*u.ambient.x;
    var color=(ambient+v.base*grain*nl*(1.0-metal*0.5)*u.light.w+spec*nl*u.light.w)*1.25;
    // Glass is deliberately dark and matte enough not to veil native labels.
    if v.props.z>2.5 { color=v.base*(0.75+0.25*nl); }
    return vec4(color,1.0);
}
