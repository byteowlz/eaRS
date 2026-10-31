//! Native geometry for the seven-bar eaRS parametric animation.
//! Reference: Design Studio `ears-parametric-states-r1`, source engine hash
//! 47422f76cf36d0135d4c2482fad62358e019a6aa062bd28fef3e4be6be14d246.
use gpui::{Hsla, IntoElement, PathBuilder, Styled, canvas, point, px};
use std::f32::consts::{FRAC_1_SQRT_2, TAU};
use std::sync::OnceLock;

const VIEWBOX: f32 = 826.0;
const WIDTH: f32 = 44.28;
const LENGTHS: [f32; 7] = [255.99, 460.51, 533.54, 701.88, 533.54, 460.51, 255.99];
const CENTERS: [f32; 7] = [248.45, 303.30, 358.15, 413.00, 467.85, 522.70, 577.55];
const PROFILE: [f32; 7] = [0.48, 0.70, 0.78, 0.73, 0.72, 0.49, 0.46];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Ready,
    Listening,
    Speech,
    Processing,
    Done,
    Attention,
}

#[derive(Clone, Copy)]
struct Settings {
    amplitude: f32,
    period: f32,
    stagger: f32,
    dwell: f32,
    speed: f32,
    direction: f32,
    mirror: bool,
    sine: bool,
}

impl State {
    fn settings(self) -> Settings {
        let mut s = Settings {
            amplitude: 0.62,
            period: 0.9,
            stagger: 0.29,
            dwell: 0.35,
            speed: 1.2,
            direction: 1.0,
            mirror: false,
            sine: true,
        };
        match self {
            Self::Ready => {
                s.amplitude = 0.10;
                s.period = 4.0;
                s.stagger = 0.10;
                s.dwell = 0.45;
                s.speed = 1.0;
                s.sine = false;
            }
            Self::Listening => {
                s.amplitude = 0.20;
                s.period = 3.0;
                s.stagger = 0.08;
                s.dwell = 0.4;
                s.speed = 1.0;
                s.sine = false;
            }
            Self::Speech => {}
            Self::Processing => {
                s.amplitude = 0.28;
                s.period = 1.8;
                s.stagger = 0.20;
                s.dwell = 0.4;
                s.direction = -1.0;
                s.speed = 1.0;
                s.sine = false;
            }
            Self::Done => {
                s.amplitude = 0.23;
                s.period = 2.4;
                s.stagger = 0.10;
                s.dwell = 0.3;
                s.mirror = true;
                s.speed = 1.0;
                s.sine = false;
            }
            Self::Attention => {
                s.amplitude = 0.12;
                s.period = 2.4;
                s.stagger = 0.05;
                s.mirror = true;
                s.speed = 1.0;
            }
        }
        s
    }
}

#[derive(Clone, Copy, Debug)]
struct Bar {
    cx: f32,
    cy: f32,
    length: f32,
}

fn corners(center: f32, length: f32, travel: f32) -> [(f32, f32); 4] {
    let cx = center + travel * FRAC_1_SQRT_2;
    let cy = center - travel * FRAC_1_SQRT_2;
    let (hw, hl) = (WIDTH / 2.0, length / 2.0);
    [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        .map(|(x, y)| (cx + (x - y) * FRAC_1_SQRT_2, cy + (x + y) * FRAC_1_SQRT_2))
}

fn travel_cap(index: usize) -> f32 {
    let inside = |distance| {
        corners(CENTERS[index], LENGTHS[index], distance)
            .iter()
            .all(|&(x, y)| (26.0..=800.0).contains(&x) && (26.0..=800.0).contains(&y))
    };
    if !inside(0.0) {
        return 0.0;
    }
    let (mut ok, mut bad) = (0.0, 1.0);
    while inside(bad) && bad < 2000.0 {
        ok = bad;
        bad *= 2.0;
    }
    for _ in 0..22 {
        let middle = (ok + bad) / 2.0;
        if inside(middle) {
            ok = middle;
        } else {
            bad = middle;
        }
    }
    ok
}

fn wave(s: Settings, t: f32, i: usize) -> f32 {
    let period = s.period * (1.0 + 0.03 * (2.0 * (((i * 7) % 5) as f32 / 4.0) - 1.0));
    let index = if s.mirror { i.abs_diff(3) } else { i } as f32;
    let phase = t / period * s.direction + index * s.stagger;
    if s.sine {
        return (TAU * phase).sin();
    }
    let triangle = 2.0 * (2.0 * phase.rem_euclid(1.0) - 1.0).abs() - 1.0;
    let x = (triangle + 1.0) / 2.0;
    let power = 1.0 + 2.0 * s.dwell.clamp(0.0, 0.95);
    let eased = if x < 0.5 {
        (2.0 * x).powf(power) / 2.0
    } else {
        1.0 - (2.0 * (1.0 - x)).powf(power) / 2.0
    };
    eased * 2.0 - 1.0
}

fn geometry(state: State, time: f32, animate: bool) -> [Bar; 7] {
    static CAPS: OnceLock<[f32; 7]> = OnceLock::new();
    let caps = CAPS.get_or_init(|| std::array::from_fn(travel_cap));
    let settings = state.settings();
    std::array::from_fn(|i| {
        let travel = if animate {
            (settings.amplitude * LENGTHS[i] * PROFILE[i]).min(caps[i]) * wave(settings, time, i)
        } else {
            0.0
        };
        Bar {
            cx: CENTERS[i] + travel * FRAC_1_SQRT_2,
            cy: CENTERS[i] - travel * FRAC_1_SQRT_2,
            length: LENGTHS[i],
        }
    })
}

pub fn mark(color: Hsla, size: f32, state: State, elapsed: f32, animate: bool) -> impl IntoElement {
    let bars = geometry(state, elapsed * state.settings().speed, animate);
    canvas(
        |_, _, _| (),
        move |bounds, _, window, _| {
            let width: f32 = bounds.size.width.into();
            let height: f32 = bounds.size.height.into();
            for bar in bars {
                let points = [
                    (-WIDTH / 2.0, -bar.length / 2.0),
                    (WIDTH / 2.0, -bar.length / 2.0),
                    (WIDTH / 2.0, bar.length / 2.0),
                    (-WIDTH / 2.0, bar.length / 2.0),
                ]
                .map(|(x, y)| {
                    point(
                        bounds.origin.x + px((bar.cx + (x - y) * FRAC_1_SQRT_2) * width / VIEWBOX),
                        bounds.origin.y + px((bar.cy + (x + y) * FRAC_1_SQRT_2) * height / VIEWBOX),
                    )
                });
                let mut path = PathBuilder::fill();
                path.add_polygon(&points, true);
                if let Ok(path) = path.build() {
                    window.paint_path(path, color);
                }
            }
        },
    )
    .w(px(size))
    .h(px(size))
    .flex_shrink_0()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_geometry_matches_reference_engine() {
        let expected = [
            273.25955, 209.09637, 341.43732, 532.81192, 417.04453, 452.30316, 619.52608,
        ];
        for (bar, x) in geometry(State::Speech, 0.37, true).iter().zip(expected) {
            assert!((bar.cx - x).abs() < 0.002, "{} != {}", bar.cx, x);
        }
    }
    #[test]
    fn other_states_match_reference_engine() {
        let expected = [
            (State::Ready, 383.87828),
            (State::Listening, 364.37288),
            (State::Processing, 334.20439),
            (State::Done, 456.93988),
            (State::Attention, 449.18925),
        ];
        for (state, center) in expected {
            let actual = geometry(state, 0.37, true)[3].cx;
            assert!(
                (actual - center).abs() < 0.002,
                "{state:?}: {actual} != {center}"
            );
        }
    }
    #[test]
    fn reduced_motion_is_exact_rest_for_every_state() {
        for state in [
            State::Ready,
            State::Listening,
            State::Speech,
            State::Processing,
            State::Done,
            State::Attention,
        ] {
            for (i, bar) in geometry(state, 3.1, false).iter().enumerate() {
                assert_eq!(
                    (bar.cx, bar.cy, bar.length),
                    (CENTERS[i], CENTERS[i], LENGTHS[i])
                );
            }
        }
    }
    #[test]
    fn state_motion_is_distinct_and_bounded() {
        let speech = geometry(State::Speech, 0.37, true);
        let listening = geometry(State::Listening, 0.37, true);
        assert!((speech[1].cx - listening[1].cx).abs() > 10.0);
        for state in [
            State::Ready,
            State::Listening,
            State::Speech,
            State::Processing,
            State::Done,
            State::Attention,
        ] {
            for tick in 0..30 {
                for (i, b) in geometry(state, tick as f32 / 7.0, true).iter().enumerate() {
                    let travel = (b.cx - CENTERS[i]) / FRAC_1_SQRT_2;
                    for (x, y) in corners(CENTERS[i], b.length, travel) {
                        assert!((25.99..=800.01).contains(&x) && (25.99..=800.01).contains(&y));
                    }
                }
            }
        }
    }
}
