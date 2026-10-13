//! Planet model + software renderer. Standard library only (no external crates).
//!
//! Fixed dense core + translucent layers that ripple outward with the music.
//! Grains are rasterised into Braille sub-cells (2x4 dots per terminal cell)
//! with a per-cell z-buffer, then emitted as ANSI rows.

use std::collections::VecDeque;
use std::io::Write;

pub const BANDS: usize = 16;
pub const K: usize = 7;
const DF: usize = 1;
const NC: usize = 360;
const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Default)]
pub struct AudioFrame {
    pub level: f32,
    pub bass: f32,
    pub bass_transient: f32,
    pub mid: f32,
    pub high: f32,
    pub bands: [f32; BANDS],
    pub beat_id: u64,
    pub beat_str: f32,
    pub density: f32,
    pub intimacy: f32,
    pub tension: f32,
    pub drop_burst: f32,
}

// ------------------------------------------------------------------------- //
// Deterministic RNG (splitmix64) - no `rand` dependency
// ------------------------------------------------------------------------- //
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn f32(&mut self) -> f32 {
        self.f64() as f32
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// n evenly spread points on a unit sphere, randomly oriented.
fn shell(n: usize, rng: &mut Rng) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let off = rng.f64() * std::f64::consts::TAU;
    // random rotation via a random unit quaternion
    let (mut qw, mut qx, mut qy, mut qz) = (rng.normal(), rng.normal(), rng.normal(), rng.normal());
    let nq = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt().max(1e-12);
    qw /= nq;
    qx /= nq;
    qy /= nq;
    qz /= nq;
    let r = [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ];
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let rad = (1.0 - y * y).max(0.0).sqrt();
            let a = golden * i as f64 + off;
            let p = [a.cos() * rad, y, a.sin() * rad];
            [
                r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
                r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
                r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
            ]
        })
        .collect()
}

// ------------------------------------------------------------------------- //
// Themes
// ------------------------------------------------------------------------- //
pub const THEME_NAMES: [&str; 4] = ["aurora", "spectrum", "ember", "matrix"];

fn theme_stops(i: usize) -> &'static [(f32, f32, f32)] {
    match i {
        0 => &[(70., 244., 255.), (40., 153., 255.), (74., 75., 255.), (70., 244., 255.)],
        1 => &[
            (44., 235., 224.), (93., 230., 255.), (255., 207., 93.), (255., 115., 125.),
            (222., 112., 255.), (105., 123., 255.), (44., 235., 224.),
        ],
        2 => &[(255., 80., 40.), (255., 150., 40.), (255., 214., 102.), (255., 90., 90.), (255., 80., 40.)],
        _ => &[(20., 200., 90.), (120., 255., 160.), (10., 140., 70.), (20., 200., 90.)],
    }
}

fn build_lut(stops: &[(f32, f32, f32)]) -> Vec<[f32; 3]> {
    (0..256)
        .map(|i| {
            let x = i as f32 / 255.0;
            let pos = x * (stops.len() - 1) as f32;
            let j = (pos.floor() as usize).min(stops.len() - 2);
            let f = pos - j as f32;
            let (a, b) = (stops[j], stops[j + 1]);
            [a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f]
        })
        .collect()
}

// ------------------------------------------------------------------------- //
// Demo audio: 32 s arc  (solo vocal -> groove -> build-up -> drop)
// ------------------------------------------------------------------------- //
pub fn demo_audio(t: f32) -> AudioFrame {
    let loop_t = t.rem_euclid(32.0);
    let beat_t = loop_t * (128.0 / 60.0);
    let beat_idx = beat_t.floor() as u64;
    let ph = beat_t - beat_t.floor();

    let mut f = AudioFrame::default();
    let mut lvl;
    if loop_t < 8.0 {
        let vp = 0.5 + 0.5 * (loop_t * 1.4).sin();
        let v = (4.0 + vp * 3.0) as usize;
        lvl = 0.24 + 0.10 * (loop_t * 2.2).sin();
        f.bands[v] = 0.85;
        f.bands[v - 1] = 0.45;
        f.bands[v + 1] = 0.40;
        f.bass = 0.15;
        f.mid = 0.70;
        f.high = 0.20;
        f.density = 0.20;
        f.intimacy = 0.85;
    } else if loop_t < 16.0 {
        let kick = (-ph * 8.0).exp();
        let sub = 0.5 + 0.5 * (beat_t * 4.0 * TAU).sin();
        let hat = (-(ph - 0.5).abs() * 12.0).exp();
        lvl = 0.45 + 0.35 * kick;
        for b in 0..3 {
            f.bands[b] = (kick * 0.85 + sub * 0.30).clamp(0.0, 1.0);
        }
        for j in 0..5 {
            f.bands[3 + j] = 0.35 * (loop_t * 2.0 + j as f32).sin().abs();
        }
        for b in 8..16 {
            f.bands[b] = hat * 0.60;
        }
        f.bass = (kick * 0.90 + sub * 0.35).clamp(0.0, 1.0);
        f.bass_transient = kick * 1.5;
        f.mid = 0.35;
        f.high = hat * 0.70;
        f.beat_str = if kick > 0.5 { 0.85 } else { 0.0 };
        f.density = 0.50;
    } else if loop_t < 24.0 {
        let bp = (loop_t - 16.0) / 8.0;
        let roll_rate = 1.0 + bp * 7.0;
        let roll_ph = (beat_t * roll_rate).rem_euclid(1.0);
        let snare = (-roll_ph * 6.0).exp();
        let riser = (bp * 14.0).clamp(2.0, 15.0) as usize;
        lvl = 0.35 + 0.50 * bp + 0.15 * snare;
        f.bands[riser] = 0.90;
        f.bands[riser.saturating_sub(1)] = 0.60;
        for b in 4..9 {
            f.bands[b] += snare * 0.70;
        }
        f.bass = 0.25 * bp;
        f.mid = (snare * 0.85 + 0.30).clamp(0.0, 1.0);
        f.high = (0.45 + 0.40 * bp).clamp(0.0, 1.0);
        f.beat_str = if snare > 0.6 { 0.75 } else { 0.0 };
        f.density = 0.45 + 0.20 * bp;
        f.tension = bp;
    } else {
        let el = loop_t - 24.0;
        let kick = (-ph * 9.0).exp();
        let clap = if beat_idx % 2 == 1 { (-(ph - 0.02).max(0.0) * 12.0).exp() } else { 0.0 };
        let sub = 0.6 + 0.4 * (beat_t * 4.0 * TAU).sin();
        let hat = (-(ph - 0.5).abs() * 14.0).exp();
        lvl = (0.65 + 0.35 * kick).clamp(0.0, 1.0);
        for b in 0..3 {
            f.bands[b] = (kick + sub * 0.5).clamp(0.0, 1.0);
        }
        for b in 3..9 {
            f.bands[b] = (clap * 0.90 + 0.60).clamp(0.0, 1.0);
        }
        for b in 9..16 {
            f.bands[b] = (hat * 0.85 + 0.50).clamp(0.0, 1.0);
        }
        f.bass = (kick * 1.0 + sub * 0.45).clamp(0.0, 1.0);
        f.bass_transient = kick * 2.0;
        f.mid = (clap * 0.90 + 0.55).clamp(0.0, 1.0);
        f.high = (hat * 0.85 + 0.50).clamp(0.0, 1.0);
        f.beat_str = if kick > 0.5 { 1.0 } else { 0.0 };
        f.density = 0.95;
        f.drop_burst = (-el * 0.6).exp();
    }
    lvl = lvl.clamp(0.0, 1.0);
    f.level = lvl;
    f.beat_id = beat_idx;
    f
}

// ------------------------------------------------------------------------- //
// Grains (struct of arrays, shuffled once so any prefix is a uniform sample)
// ------------------------------------------------------------------------- //
struct Grains {
    dx: Vec<f32>,
    dy: Vec<f32>,
    dz: Vec<f32>,
    phase: Vec<f32>,
    freq: Vec<f32>,
    thick: Vec<f32>,
    theta_lat: Vec<f32>,
    tidal: Vec<f32>,
    id_exp: Vec<f32>,
    alpha_base: Vec<f32>,
    cpos_base: Vec<f32>,
    zphase: Vec<f32>,
    sid: Vec<u8>,
    band: Vec<u8>,
}

impl Grains {
    fn new(rng: &mut Rng) -> Self {
        let mut raw: Vec<([f64; 3], u8)> = Vec::new();
        for k in 0..K {
            for p in shell(900 + 250 * k, rng) {
                raw.push((p, k as u8));
            }
        }
        let m = raw.len();
        let mut idx: Vec<usize> = (0..m).collect();
        for i in (1..m).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut g = Grains {
            dx: Vec::with_capacity(m), dy: Vec::with_capacity(m), dz: Vec::with_capacity(m),
            phase: Vec::with_capacity(m), freq: Vec::with_capacity(m), thick: Vec::with_capacity(m),
            theta_lat: Vec::with_capacity(m), tidal: Vec::with_capacity(m), id_exp: Vec::with_capacity(m),
            alpha_base: Vec::with_capacity(m), cpos_base: Vec::with_capacity(m), zphase: Vec::with_capacity(m),
            sid: Vec::with_capacity(m), band: Vec::with_capacity(m),
        };
        for &i in &idx {
            let (p, k) = raw[i];
            let (x, y, z) = (p[0] as f32, p[1] as f32, p[2] as f32);
            g.dx.push(x);
            g.dy.push(y);
            g.dz.push(z);
            let phase = rng.f32() * TAU;
            g.phase.push(phase);
            g.freq.push(0.8 + rng.f32() * 1.6);
            g.thick.push(rng.normal() as f32);
            let th = y.clamp(-1.0, 1.0).acos();
            g.theta_lat.push(th);
            g.tidal.push(2.5 * y * y - 0.8);
            g.id_exp.push((-(k as f32) * 0.25).exp());
            g.alpha_base.push(0.70 + 0.03 * k as f32);
            g.cpos_base.push(k as f32 / (K - 1) as f32 * 0.70);
            g.zphase.push(phase / TAU);
            g.sid.push(k);
            g.band.push(((th / std::f32::consts::PI * BANDS as f32) as usize).min(BANDS - 1) as u8);
        }
        g
    }
}

const BITS: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

// ------------------------------------------------------------------------- //
// Planet
// ------------------------------------------------------------------------- //
pub struct Planet {
    g: Grains,
    pub m: usize,
    pub n: usize,
    pub quality: f32,
    pub theme: usize,
    pub spin_scale: f32,

    c_g: Vec<[f32; 2]>,
    c_fall: Vec<f32>,
    luts: Vec<Vec<[f32; 3]>>,
    sgr: Vec<Vec<u8>>,

    base: [f32; K],
    maxf: [f32; K],
    kk: [f32; K],
    cc: [f32; K],
    speeds: [f32; K],

    lvl_fast: f32,
    lvl_slow: f32,
    pulse: f32,
    bands: [f32; BANDS],
    hist: VecDeque<(f32, f32, f32, f32)>,
    r: [f32; K],
    rv: [f32; K],
    lk: [f32; K],
    pk: [f32; K],
    bk: [f32; K],
    btk: [f32; K],
    pub high: f32,
    pub brightness: f32,
    density: f32,
    intimacy: f32,
    tension: f32,
    drop_burst: f32,
    intensity: f32,
    layer_yaws: [f32; K],
    color_phase: f32,
    last_bid: u64,

    // raster buffers
    mask: Vec<u8>,
    zbuf: Vec<f32>,
    key: Vec<u16>,
    pub rows: Vec<Vec<u8>>,
}

impl Planet {
    pub fn new() -> Self {
        let mut rng = Rng::new(7);
        let g = Grains::new(&mut rng);
        let m = g.dx.len();
        let c_g: Vec<[f32; 2]> = (0..NC).map(|_| [rng.normal() as f32, rng.normal() as f32]).collect();
        let c_fall = c_g
            .iter()
            .map(|p| 1.0 - 0.50 * ((p[0] * p[0] + p[1] * p[1]).sqrt() / 2.5).min(1.0))
            .collect();
        let luts = (0..4).map(|i| build_lut(theme_stops(i))).collect();
        let sgr = (0..4096u32)
            .map(|k| {
                let r = ((k >> 8) & 15) << 4 | 8;
                let g = ((k >> 4) & 15) << 4 | 8;
                let b = (k & 15) << 4 | 8;
                format!("\x1b[38;2;{r};{g};{b}m").into_bytes()
            })
            .collect();

        let mut base = [0.0; K];
        let mut maxf = [0.0; K];
        let mut kk = [0.0; K];
        let mut cc = [0.0; K];
        for k in 0..K {
            base[k] = 0.15 + 0.028 * k as f32;
            maxf[k] = 0.32 + 0.65 * (k as f32 / (K - 1) as f32).powf(0.85);
            kk[k] = 260.0 - 15.0 * k as f32;
            cc[k] = 2.0 * 0.58 * kk[k].sqrt();
        }
        Planet {
            g, m, n: m, quality: 1.0, theme: 0, spin_scale: 1.0,
            c_g, c_fall, luts, sgr,
            base, maxf, kk, cc,
            speeds: [1.5, -1.1, 0.85, -0.65, 0.50, -0.40, 0.32],
            lvl_fast: 0.0, lvl_slow: 0.0, pulse: 0.0, bands: [0.0; BANDS],
            hist: VecDeque::new(), r: base, rv: [0.0; K], lk: [0.0; K], pk: [0.0; K],
            bk: [0.0; K], btk: [0.0; K], high: 0.0, brightness: 1.35,
            density: 0.0, intimacy: 0.0, tension: 0.0, drop_burst: 0.0, intensity: 0.0,
            layer_yaws: [0.0; K], color_phase: 0.0, last_bid: 0,
            mask: Vec::new(), zbuf: Vec::new(), key: Vec::new(), rows: Vec::new(),
        }
    }

    pub fn set_quality(&mut self, q: f32) {
        self.quality = q.clamp(0.30, 1.0);
        self.n = (self.m as f32 * self.quality) as usize;
    }

    /// Advance the simulation by `dt_raw` seconds with the current audio frame.
    pub fn step(&mut self, dt_raw: f32, a: &AudioFrame) {
        let dt = dt_raw.clamp(0.0001, 0.035);
        let follow = |cur: &mut f32, target: f32, tau: f32| *cur += (target - *cur) * (1.0 - (-dt / tau).exp());

        follow(&mut self.density, a.density, 0.06);
        follow(&mut self.intimacy, a.intimacy, 0.06);
        follow(&mut self.tension, a.tension, 0.06);
        self.drop_burst = a.drop_burst.max(self.drop_burst * (-dt / 0.8).exp());
        self.high = a.high;

        if a.beat_id != self.last_bid {
            self.last_bid = a.beat_id;
            self.pulse = 1.0;
            let kick = 2.4 * a.beat_str;
            let pop = 0.05 * a.beat_str;
            for k in 0..K {
                self.rv[k] += kick * (-(k as f32) * 0.30).exp();
                self.r[k] += pop * (-(k as f32) * 0.25).exp();
            }
        }
        // Continuous bass transient response: sharp bass attacks impart immediate velocity
        if a.bass_transient > 0.35 {
            let kick_trans = (a.bass_transient - 0.35) * 1.5;
            for k in 0..3 {
                self.rv[k] += kick_trans * (-(k as f32) * 0.40).exp();
            }
        }
        self.pulse *= (-dt / 0.14).exp();

        // Instant attack for zero latency / immediate reaction
        let lv = a.level;
        if lv > self.lvl_fast {
            self.lvl_fast = lv;
        } else {
            follow(&mut self.lvl_fast, lv, 0.04);
        }
        if lv > self.lvl_slow {
            self.lvl_slow += (lv - self.lvl_slow) * 0.85;
        } else {
            follow(&mut self.lvl_slow, lv, 0.15);
        }

        let down_k = 1.0 - (-dt / 0.06).exp();
        for i in 0..BANDS {
            if a.bands[i] > self.bands[i] {
                self.bands[i] = a.bands[i];
            } else {
                self.bands[i] += (a.bands[i] - self.bands[i]) * down_k;
            }
        }

        self.intensity = (self.lvl_fast * 0.65 + a.bass * 0.55 + 0.35 * self.pulse).min(1.0);

        let spin_mult = self.spin_scale
            * (0.65 + 1.2 * self.lvl_fast + 1.2 * a.bass + 1.5 * self.density + 1.8 * self.pulse + 2.5 * self.drop_burst);
        for k in 0..K {
            self.layer_yaws[k] += dt * self.speeds[k] * spin_mult;
            // keep the angle bounded so f32 sin/cos stay precise over hours
            self.layer_yaws[k] = self.layer_yaws[k].rem_euclid(TAU);
        }
        self.color_phase = (self.color_phase
            + dt * (0.04 + 0.12 * self.lvl_fast + 0.18 * a.bass + 0.22 * self.pulse + 0.30 * self.drop_burst))
            .rem_euclid(1.0);

        // Acoustic delay line: wave travels outward from core through layers (k * DF frames delay)
        self.hist.push_front((self.lvl_fast, self.pulse, a.bass, a.bass_transient));
        self.hist.truncate(K * DF + 1);
        let nh = self.hist.len();
        for k in 0..K {
            let (l, p, b, bt) = self.hist[(k * DF).min(nh - 1)];
            self.lk[k] = l.min(1.0).powf(0.85);
            self.pk[k] = p;
            self.bk[k] = b;
            self.btk[k] = bt;
        }

        // Spring integration: acoustic pressure expands shells smoothly with the rhythm
        let h = dt / 2.0;
        for _ in 0..2 {
            for k in 0..K {
                let bass_pressure = self.bk[k] * 0.60 * (0.6 + 0.4 * k as f32 / (K - 1) as f32);
                let ripple = 0.16 * self.pk[k] * (0.5 + 0.5 * k as f32 / (K - 1) as f32);
                let fullness = 0.18 * self.lk[k] + bass_pressure + 0.25 * self.density + 0.20 * self.drop_burst;
                let reach = self.maxf[k] * (0.90 + 0.25 * self.drop_burst);
                let target = self.base[k] + (reach - self.base[k]) * fullness.min(1.1) + ripple;
                let acc = self.kk[k] * (target - self.r[k]) - self.cc[k] * self.rv[k];
                self.rv[k] += acc * h;
                self.r[k] += self.rv[k] * h;
            }
        }
    }

    /// Rasterise into `self.rows` (one ANSI string per terminal row, no cursor moves).
    pub fn render(&mut self, w: usize, h: usize, t: f32) {
        let (ww, hh) = (w * 2, h * 4);
        let cells = w * h;
        if self.mask.len() != cells {
            self.mask = vec![0; cells];
            self.zbuf = vec![f32::NEG_INFINITY; cells];
            self.key = vec![0; cells];
        } else {
            self.mask.fill(0);
            self.zbuf.fill(f32::NEG_INFINITY);
        }
        let cxs = (ww - 1) as f32 / 2.0;
        let cys = (hh - 1) as f32 / 2.0;
        let rmax = (cxs.min(cys) - 1.0).max(4.0);

        let singularity = (0.55 * self.density + 0.65 * self.drop_burst + 0.40 * self.intensity).clamp(0.0, 1.0);
        let sigma = 0.22 + (1.0 - singularity).powf(1.8) * (0.65 + 0.025 * rmax);
        let clear = 1.8 + 2.0 * sigma;
        let clear_sq = clear * clear;

        // per-frame constants
        let pitch = 0.30 + 0.06 * (t * 0.45).sin();
        let (sp, cp) = pitch.sin_cos();
        let mut cyk = [0f32; K];
        let mut syk = [0f32; K];
        let mut spread = [0f32; K];
        let mut alpha_k = [0f32; K];
        for k in 0..K {
            let (s, c) = self.layer_yaws[k].sin_cos();
            syk[k] = s;
            cyk[k] = c;
            spread[k] = (self.r[k] - self.base[k]).max(0.0);
            alpha_k[k] = 0.35 * self.lk[k] + 0.90 * self.pk[k] + 0.45 * self.bk[k];
        }
        let stretch_c = 0.38 * self.tension;
        let trem_c = self.tension * 0.040;
        let voc_c = 0.09 * self.intimacy;
        let pulse_c = 0.12 * self.pulse;
        let thick_c = 0.020 + 0.045 * self.density;
        let tt38 = t * 38.0;
        let tt45 = t * 4.5;
        let tt5 = t * 5.0;
        let (lx, ly, lz) = (-0.38f32, -0.55f32, 0.74f32);
        let lut = &self.luts[self.theme];
        let cphase = self.color_phase;
        let d_cam = 2.7f32;

        let g = &self.g;
        let n = self.n;
        for i in 0..n {
            let k = g.sid[i] as usize;
            let (x, y, z) = (g.dx[i], g.dy[i], g.dz[i]);
            let x1 = x * cyk[k] - z * syk[k];
            let z1 = x * syk[k] + z * cyk[k];
            let px = x1;
            let py = y * cp - z1 * sp;
            let pz = y * sp + z1 * cp;

            let bnd = self.bands[g.band[i] as usize];
            let ph = g.phase[i];
            let stretch = 1.0 + stretch_c * g.tidal[i];
            let tremble = trem_c * (tt38 + ph).sin();
            let vocal = voc_c * (3.0 * g.theta_lat[i] + tt45).cos() * bnd;
            let wob = 0.025 * (g.freq[i] * tt5 + ph).sin();

            // Sound wave mathematics:
            // 1. Acoustic standing wave ripple (Chladni pattern) across spherical harmonics
            let standing_wave = 0.06 * (self.bk[k] * 0.6 + bnd * 0.7) * (4.0 * g.theta_lat[i] + tt38 + ph).sin();
            // 2. High-frequency treble shimmer
            let treble_shimmer = 0.04 * self.high * (8.0 * g.theta_lat[i] + tt45 * 2.0).cos();
            // 3. Band-specific vibrational displacement
            let band_disp = 0.28 * bnd;

            let mut rr = self.r[k] * stretch * (1.0 + wob + tremble + vocal + band_disp + standing_wave + treble_shimmer + pulse_c * g.id_exp[i])
                + g.thick[i] * (0.003 + thick_c * spread[k]);
            if rr < 0.0 {
                rr = 0.0;
            }
            if rr > 0.94 {
                rr = 0.94 + 0.06 * ((rr - 0.94) / 0.06).tanh();
            }

            let persp = d_cam / (d_cam - pz * rr).max(0.4);
            let scale = rr * rmax * persp;
            let sx = (px * scale + cxs).round() as i32;
            let sy = (py * scale + cys).round() as i32;
            if (sx as u32) >= ww as u32 || (sy as u32) >= hh as u32 {
                continue;
            }

            let dx = sx as f32 - cxs;
            let dy = sy as f32 - cys;
            if !(pz > 0.10 || dx * dx + dy * dy > clear_sq) {
                continue;
            }
            let zf = ((pz + 1.0) * 0.5).clamp(0.0, 1.0);
            if !(pz > -0.15 || g.zphase[i] < 0.30 + 0.70 * zf) {
                continue;
            }

            // Perceptual brightness and illumination:
            let diff = (px * lx + py * ly + pz * lz).max(0.0);
            let spec = diff.powi(6);
            let fog = 0.50 + 0.50 * zf;
            let rim = ((1.0 - pz.abs()).powf(2.0) * 0.50).clamp(0.0, 0.50);
            let shade = (0.35 + 0.45 * diff + 0.55 * spec + rim) * fog;

            // Audio glow + user brightness multiplier:
            let audio_glow = 0.35 * self.lvl_fast + 0.50 * self.pulse + 0.30 * self.bk[k];
            let bright = ((g.alpha_base[i] + alpha_k[k] + audio_glow) * shade * self.brightness).clamp(0.0, 1.50);

            let cpos = (g.cpos_base[i] + cphase).rem_euclid(1.0);
            let c = lut[(cpos * 255.0) as usize];

            // Neon bloom: on loud beats and specular highlights, blend towards white-hot shine
            let over = (bright - 1.0).max(0.0);
            let r = (c[0] * bright.min(1.0) + 255.0 * over * 0.85).clamp(0.0, 255.0);
            let g = (c[1] * bright.min(1.0) + 255.0 * over * 0.85).clamp(0.0, 255.0);
            let b = (c[2] * bright.min(1.0) + 255.0 * over * 0.85).clamp(0.0, 255.0);
            let key = pack(r, g, b);

            let cell = (sy as usize >> 2) * w + (sx as usize >> 1);
            self.mask[cell] |= BITS[(sy & 3) as usize][(sx & 1) as usize];
            if pz >= self.zbuf[cell] {
                self.zbuf[cell] = pz;
                self.key[cell] = key;
            }
        }

        // ---- core: tiny spinning nucleus at z = 0.02 ----
        let core_spin = t * (2.5 + 5.0 * self.pulse + 8.0 * self.drop_burst);
        let (sn, cs) = core_spin.rem_euclid(TAU).sin_cos();
        let white = (0.25 + 0.75 * singularity + 0.30 * self.pulse).min(1.0);
        let core_power = (0.85 + 0.35 * singularity + 0.50 * self.pulse) * self.brightness;
        let l35 = lut[35];
        let warm = [
            l35[0] * (1.0 - white) + 255.0 * white,
            l35[1] * (1.0 - white) + 255.0 * white,
            l35[2] * (1.0 - white) + 255.0 * white,
        ];
        for j in 0..NC {
            let gx = self.c_g[j][0] * cs - self.c_g[j][1] * sn;
            let gy = self.c_g[j][0] * sn + self.c_g[j][1] * cs;
            let sx = ((cxs + 0.5 + gx * sigma).floor() as i32).clamp(0, ww as i32 - 1) as usize;
            let sy = ((cys + 0.5 + gy * sigma).floor() as i32).clamp(0, hh as i32 - 1) as usize;
            let cb = (core_power * self.c_fall[j]).clamp(0.0, 1.5);
            let over = (cb - 1.0).max(0.0);
            let r = (warm[0] * cb.min(1.0) + 255.0 * over * 0.9).clamp(0.0, 255.0);
            let g = (warm[1] * cb.min(1.0) + 255.0 * over * 0.9).clamp(0.0, 255.0);
            let b = (warm[2] * cb.min(1.0) + 255.0 * over * 0.9).clamp(0.0, 255.0);
            let key = pack(r, g, b);
            let cell = (sy >> 2) * w + (sx >> 1);
            self.mask[cell] |= BITS[sy & 3][sx & 1];
            if 0.02 >= self.zbuf[cell] {
                self.zbuf[cell] = 0.02;
                self.key[cell] = key;
            }
        }

        self.build_rows(w, h);
    }

    fn build_rows(&mut self, w: usize, h: usize) {
        if self.rows.len() != h {
            self.rows = vec![Vec::new(); h];
        }
        for r in 0..h {
            let row = &mut self.rows[r];
            row.clear();
            let base = r * w;
            let mut last_col = 0usize;
            let mut last_key = u16::MAX;
            for c in 0..w {
                let m = self.mask[base + c];
                if m == 0 {
                    continue;
                }
                let key = self.key[base + c];
                if c > last_col {
                    let gap = c - last_col;
                    if gap < 5 {
                        row.extend(std::iter::repeat(b' ').take(gap));
                    } else {
                        let _ = write!(row, "\x1b[{}G", c + 1);
                    }
                }
                if key != last_key {
                    row.extend_from_slice(&self.sgr[key as usize]);
                    last_key = key;
                }
                // U+2800 + m, UTF-8 encoded (E2 A0..A3 80..BF)
                row.extend_from_slice(&[0xE2, 0xA0 + (m >> 6), 0x80 + (m & 0x3F)]);
                last_col = c + 1;
            }
        }
    }
}

#[inline]
fn pack(r: f32, g: f32, b: f32) -> u16 {
    let r = (r.clamp(0.0, 255.0) as u32) >> 4;
    let g = (g.clamp(0.0, 255.0) as u32) >> 4;
    let b = (b.clamp(0.0, 255.0) as u32) >> 4;
    ((r << 8) | (g << 4) | b) as u16
}
