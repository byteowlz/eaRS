//! Portable offscreen renderer. All I/O is caller-owned; run on a worker, never the UI thread.
mod matrix;
mod mesh;
use bytemuck::{Pod, Zeroable};
use device_scene::*;
pub use mesh::{HitTriangle, Label, TextRaster};
use mesh::{Mesh, Vertex};
use std::sync::mpsc;
use std::time::Duration;
use wgpu::util::DeviceExt;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("{0}")]
    Invalid(#[from] ValidationError),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error("render budget: {0}")]
    Budget(&'static str),
}
/// A complete, indivisible visual result. Hit geometry is the same triangles submitted to wgpu.
#[derive(Debug)]
pub struct RenderFrame {
    pub request_generation: u64,
    pub scene_id: String,
    pub scene_revision: u64,
    pub state_revision: u64,
    pub viewport: Viewport,
    pub projection: Projection,
    pub width_px: u32,
    pub height_px: u32,
    pub stride_bytes: u32,
    pub pixels: Vec<u8>,
    pub hit_geometry: Vec<HitTriangle>,
    pub labels: Vec<Label>,
}
impl RenderFrame {
    /// A non-control surface occludes controls behind it. Disabled controls are filtered by the host.
    pub fn hit_test(&self, x_dip: f64, y_dip: f64) -> Option<&str> {
        self.hit_surface(x_dip, y_dip)?.control_id.as_deref()
    }

    /// Only visible non-control geometry is draggable, never empty canvas or a disabled control.
    pub fn is_chassis_at(&self, x_dip: f64, y_dip: f64) -> bool {
        self.hit_surface(x_dip, y_dip)
            .is_some_and(|surface| surface.control_id.is_none())
    }

    fn hit_surface(&self, x_dip: f64, y_dip: f64) -> Option<&HitTriangle> {
        let ray = self.projection.ray(x_dip, y_dip)?;
        self.hit_geometry
            .iter()
            .filter_map(|t| ray.triangle(t.vertices).map(|distance| (distance, t)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, t)| t)
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    vp: [[f32; 4]; 4],
    eye: [f32; 4],
    light: [f32; 4],
    ambient: [f32; 4],
}
struct Targets {
    width: u32,
    height: u32,
    row_pitch: u32,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _msaa: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    readback: wgpu::Buffer,
}
pub struct OffscreenRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    targets: Option<Targets>,
    pub adapter_name: String,
}
impl OffscreenRenderer {
    pub async fn new() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError::Gpu(e.to_string()))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device proof"),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError::Gpu(e.to_string()))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native receiver material"),
            source: wgpu::ShaderSource::Wgsl(include_str!("material.wgsl").into()),
        });
        let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32x3,3=>Float32x3,4=>Float32x2,5=>Float32x3];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("receiver"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("projection and light"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            uniforms,
            bind,
            targets: None,
            adapter_name,
        })
    }
    /// Synchronous GPU submission/readback. Only a renderer worker may call this.
    /// Persistent target budget: 4,194,304 pixels, one color/MSAA/depth/staging set.
    pub fn render(
        &mut self,
        snapshot: &ValidatedSnapshot,
        generation: u64,
    ) -> Result<RenderFrame, RenderError> {
        let projection = Projection::new(snapshot)?;
        let e = projection.extent;
        if u64::from(e.width_px) * u64::from(e.height_px) > 4_194_304 {
            return Err(RenderError::Budget("proof maximum is 4,194,304 pixels"));
        }
        if e.width_px > self.device.limits().max_texture_dimension_2d
            || e.height_px > self.device.limits().max_texture_dimension_2d
        {
            return Err(RenderError::Budget(
                "texture dimension exceeds adapter limit",
            ));
        }
        let mesh = Mesh::build(snapshot)?;
        let s = snapshot.snapshot();
        let az = s.scene.light.azimuth_deg.to_radians();
        let el = s.scene.light.elevation_deg.to_radians();
        let uniforms = Uniforms {
            vp: projection.view_projection.as_mat4().to_cols_array_2d(),
            eye: projection.eye.as_vec3().extend(1.0).to_array(),
            light: [
                (el.cos() * az.sin()) as f32,
                el.sin() as f32,
                (el.cos() * az.cos()) as f32,
                s.scene.light.intensity as f32,
            ],
            ambient: [s.scene.light.ambient as f32, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        if self
            .targets
            .as_ref()
            .is_none_or(|t| t.width != e.width_px || t.height != e.height_px)
        {
            // Release old resources before allocation, including on resize storms.
            self.targets = None;
            self.targets = Some(self.allocate(e.width_px, e.height_px));
        }
        let t = self.targets.as_ref().expect("allocated targets");
        let vertices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bounded receiver mesh"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("receiver frame"),
            });
        {
            let bg = if s.viewport.transparent {
                wgpu::Color::TRANSPARENT
            } else {
                wgpu::Color {
                    r: 0.008,
                    g: 0.010,
                    b: 0.012,
                    a: 1.0,
                }
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("receiver"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&t.color_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.draw(0..mesh.vertices.len() as u32, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &t.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &t.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(t.row_pitch),
                    rows_per_image: Some(t.height),
                },
            },
            wgpu::Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: 1,
            },
        );
        let submission = self.queue.submit([encoder.finish()]);
        let pixels = self.read_pixels(t, submission)?;
        Ok(RenderFrame {
            request_generation: generation,
            scene_id: s.scene.id.clone(),
            scene_revision: s.scene.revision,
            state_revision: s.state.revision,
            viewport: s.viewport.clone(),
            width_px: e.width_px,
            height_px: e.height_px,
            stride_bytes: e.width_px * 4,
            projection,
            pixels,
            hit_geometry: mesh.hits,
            labels: mesh.labels,
        })
    }
    fn read_pixels(
        &self,
        target: &Targets,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Vec<u8>, RenderError> {
        let (tx, rx) = mpsc::sync_channel(1);
        target
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        // Always unmap, even after timeout, callback failure or malformed readback.
        let result = (|| {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|e| RenderError::Gpu(e.to_string()))?;
            rx.recv_timeout(Duration::from_secs(1))
                .map_err(|e| RenderError::Gpu(e.to_string()))?
                .map_err(|e| RenderError::Gpu(e.to_string()))?;
            let mapped = target
                .readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| RenderError::Gpu(e.to_string()))?;
            bgra_from_padded_srgb(&mapped, target.width, target.height, target.row_pitch)
        })();
        target.readback.unmap();
        result
    }
    fn allocate(&self, width: u32, height: u32) -> Targets {
        let texture = |label, format, count, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = texture(
            "readback color",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let msaa = texture(
            "4x geometry coverage",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth = texture(
            "occlusion",
            wgpu::TextureFormat::Depth32Float,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let row_pitch = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("padded readback"),
            size: u64::from(row_pitch) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Targets {
            width,
            height,
            row_pitch,
            color_view: color.create_view(&Default::default()),
            msaa_view: msaa.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            color,
            _msaa: msaa,
            _depth: depth,
            readback,
        }
    }
}
/// Unpad RGBA sRGB MSAA output and convert linear-premultiplied coverage to
/// encoded-sRGB-premultiplied BGRA, as required by device-scene v1.
pub fn bgra_from_padded_srgb(
    data: &[u8],
    width: u32,
    height: u32,
    row_pitch: u32,
) -> Result<Vec<u8>, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::Budget("zero readback extent"));
    }
    let packed = width
        .checked_mul(4)
        .ok_or(RenderError::Budget("row overflow"))?;
    if row_pitch < packed || u64::from(row_pitch) * u64::from(height) > data.len() as u64 {
        return Err(RenderError::Budget("invalid padded readback"));
    }
    let mut out = Vec::with_capacity(packed as usize * height as usize);
    for row in data.chunks_exact(row_pitch as usize).take(height as usize) {
        for p in row[..packed as usize].chunks_exact(4) {
            let a = p[3] as f32 / 255.0;
            let convert = |b: u8| -> u8 {
                if p[3] == 0 {
                    return 0;
                }
                if p[3] == 255 {
                    return b;
                }
                let s = b as f32 / 255.0;
                let linear = if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                };
                let straight = (linear / a).clamp(0.0, 1.0);
                let encoded = if straight <= 0.0031308 {
                    straight * 12.92
                } else {
                    1.055 * straight.powf(1.0 / 2.4) - 0.055
                };
                (encoded * a * 255.0).round() as u8
            };
            out.extend_from_slice(&[convert(p[2]), convert(p[1]), convert(p[0]), p[3]]);
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swaps_channels_removes_padding_and_zeros_transparent_rgb() {
        let data = [
            255, 0, 0, 255, 0, 0, 255, 255, 77, 88, 99, 0, 123, 123, 123, 123,
        ];
        assert_eq!(
            bgra_from_padded_srgb(&data, 3, 1, 16).unwrap(),
            [0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0]
        );
    }
    #[test]
    fn transparent_coverage_is_premultiplied_in_encoded_space() {
        let result = bgra_from_padded_srgb(&[188, 0, 0, 128], 1, 1, 4).unwrap();
        assert_eq!(result, [0, 0, 128, 128]);
    }
}
