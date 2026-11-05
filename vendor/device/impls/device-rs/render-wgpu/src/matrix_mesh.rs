//! Matrix phosphors live on the display plane and use its camera/depth, not GPUI text.
use super::*;

impl Mesh {
    pub(super) fn matrix_text(
        &mut self,
        text: &str,
        matrix: &Matrix,
        rect: &Rect,
        origin: [f64; 3],
        mat: &Material,
    ) -> Result<TextRaster, RenderError> {
        let Some(grid) = crate::matrix::layout(text, matrix) else {
            return Ok(TextRaster::UnicodeFallback);
        };
        let dots = matrix.cols as usize
            * matrix.rows as usize
            * matrix.glyph_width as usize
            * matrix.glyph_height as usize;
        if self.vertices.len() + dots * 6 > MAX_VERTICES {
            return Err(RenderError::Budget("matrix vertex budget exceeded"));
        }
        let start = Vec3::new(
            (origin[0] + rect.x - rect.width / 2.0 + matrix.padding_x) as f32,
            (origin[1] + rect.y + rect.height / 2.0 - matrix.padding_y) as f32,
            (origin[2] + rect.z + 0.025) as f32,
        );
        let pitch = matrix.pitch as f32;
        for (index, glyph) in grid.cells.iter().enumerate() {
            let cell_x = index as u32 % matrix.cols * matrix.cell_width;
            let cell_y = index as u32 / matrix.cols * matrix.cell_height;
            for row in 0..matrix.glyph_height {
                for col in 0..matrix.glyph_width {
                    let lit = glyph[row as usize] & (1 << (4 - col)) != 0;
                    let center = start
                        + Vec3::new(
                            (cell_x + col) as f32 * pitch + pitch / 2.0,
                            -((cell_y + row) as f32 * pitch + pitch / 2.0),
                            0.0,
                        );
                    self.matrix_dot(center, pitch, mat, lit);
                }
            }
        }
        Ok(if grid.clipped {
            TextRaster::MatrixClipped
        } else {
            TextRaster::Matrix
        })
    }

    fn matrix_dot(&mut self, center: Vec3, pitch: f32, mat: &Material, lit: bool) {
        let ink = linear_color(&mat.ink).map(|v| v * if lit { 1.0 } else { 0.006 });
        for local in [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ] {
            self.vertices.push(Vertex {
                position: (center + Vec3::new(local[0] * pitch, local[1] * pitch, 0.0)).to_array(),
                normal: [0.0, 0.0, 1.0],
                base: linear_color(&mat.base),
                props: [0.8, 0.0, 4.0],
                local,
                emission: ink,
            });
        }
        // Decorative dots do not duplicate hit triangles: the control's display surface owns input.
    }
}
