//! Shared 5x7 receiver alphabet: r10 originals plus case-preserving text glyphs.
//! Display geometry comes from the scene, never from the length of a value.
use device_scene::Matrix;
use std::{collections::BTreeMap, sync::OnceLock};

type Glyph = [u8; 8];
fn glyphs() -> &'static BTreeMap<char, Glyph> {
    static TABLE: OnceLock<BTreeMap<char, Glyph>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let source: BTreeMap<char, Vec<String>> =
            serde_json::from_str(include_str!("matrix-glyphs.json")).expect("bundled glyphs");
        source
            .into_iter()
            .map(|(ch, rows)| {
                let mut glyph = [0; 8];
                for (index, row) in rows.iter().enumerate() {
                    glyph[index] = u8::from_str_radix(row, 2).expect("binary glyph");
                }
                (ch, glyph)
            })
            .collect()
    })
}

pub(super) struct Grid {
    pub cells: Vec<Glyph>,
    pub clipped: bool,
}
/// Missing glyphs fall back as a complete native string: never tofu, transliteration,
/// forced uppercase, or splitting a Unicode grapheme into individual matrix cells.
pub(super) fn layout(text: &str, matrix: &Matrix) -> Option<Grid> {
    let font = glyphs();
    if text.chars().any(|ch| ch != '\n' && !font.contains_key(&ch)) {
        return None;
    }
    let (cols, rows) = (matrix.cols as usize, matrix.rows as usize);
    let mut cells = vec![[0; 8]; cols * rows];
    let lines: Vec<Vec<char>> = text
        .split('\n')
        .flat_map(|paragraph| wrap(paragraph, cols))
        .map(trim)
        .collect();
    // Every display centres its text: each line across, the block of lines down.
    let shown = lines.len().min(rows);
    let first = (rows - shown) / 2;
    for (i, line) in lines.iter().take(rows).enumerate() {
        let offset = (cols - line.len().min(cols)) / 2;
        for (col, ch) in line.iter().take(cols).enumerate() {
            cells[(first + i) * cols + offset + col] = font[ch];
        }
    }
    let clipped = lines.len() > matrix.rows as usize;
    if clipped {
        *cells.last_mut().expect("validated nonempty matrix") = font[&'…'];
    }
    Some(Grid { cells, clipped })
}

fn trim(mut line: Vec<char>) -> Vec<char> {
    while line.last() == Some(&' ') {
        line.pop();
    }
    let lead = line.iter().take_while(|c| **c == ' ').count();
    line.drain(..lead);
    line
}

fn wrap(text: &str, cols: usize) -> Vec<Vec<char>> {
    let chars: Vec<_> = text.chars().collect();
    let mut lines = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + cols).min(chars.len());
        if end < chars.len()
            && chars[end] != ' '
            && let Some(space) = chars[start..end].iter().rposition(|ch| *ch == ' ')
            && space > 0
        {
            end = start + space;
        }
        lines.push(chars[start..end].to_vec());
        start = end;
        while start < chars.len() && chars[start] == ' ' {
            start += 1;
        }
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    fn matrix(cols: u32, rows: u32) -> Matrix {
        Matrix {
            cols,
            rows,
            glyph_width: 5,
            glyph_height: 7,
            cell_width: 6,
            cell_height: 8,
            pitch: 0.16,
            padding_x: 0.4,
            padding_y: 0.2,
        }
    }
    #[test]
    fn glyph_table_covers_ascii_without_uppercase_substitution() {
        let font = glyphs();
        for ch in ' '..='~' {
            assert!(font.contains_key(&ch), "{ch}");
        }
        assert_ne!(font[&'a'], font[&'A']);
        assert_ne!(font[&'s'], font[&'5']);
        assert!(font.values().all(|rows| rows.iter().all(|row| *row < 32)));
        assert_eq!(&font[&'0'][..7], &[14, 17, 19, 21, 25, 17, 14]);
    }
    #[test]
    fn fixed_capacity_wraps_words_and_marks_overflow_without_changing_the_source() {
        let m = matrix(8, 2);
        let text = "Right Option";
        let grid = layout(text, &m).unwrap();
        assert_eq!(grid.cells.len(), 16);
        // "Right" / "Option", each centred in 8 columns.
        assert_eq!(grid.cells[1], glyphs()[&'R']);
        assert_eq!(grid.cells[9], glyphs()[&'O']);
        assert!(!grid.clipped);
        let long = layout("right_ctrl+shift+alt+f12", &m).unwrap();
        assert!(long.clipped);
        assert_eq!(long.cells[15], glyphs()[&'…']);
        assert_eq!(layout("A\nB", &m).unwrap().cells[11], glyphs()[&'B']);
        // A single line sits on the middle row of a three-row display.
        let three = matrix(5, 3);
        assert_eq!(layout("Hi", &three).unwrap().cells[5 + 1], glyphs()[&'H']);
        assert_eq!(text, "Right Option");
        // Device names use typographic apostrophes and accents; they stay dot matrix.
        for name in ["Tommy’s iPhone", "Jürgen’s AirPods", "Café Mic", "“Studio”"] {
            assert!(layout(name, &matrix(24, 1)).is_some(), "{name}");
        }
        assert!(layout("日本語", &m).is_none());
        assert!(layout("e\u{301}", &m).is_none());
    }
}
