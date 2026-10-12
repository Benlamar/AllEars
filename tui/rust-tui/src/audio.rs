//! System-audio loopback capture (cpal) + loudness / beat / mood analysis (rustfft).
//! Direct port of the Python `SystemAudio` class.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::planet::{AudioFrame, BANDS};

pub const BLOCK: usize = 512;

pub struct AudioShared {
    pub frame: AudioFrame,
    pub ok: bool,
    pub error: String,
    pub source: String,
    /// user gain in dB (Up/Down keys)
    pub sens_db: f32,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Audio {
    shared: Arc<Mutex<AudioShared>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Audio {
    pub fn new() -> Self {
        Audio {
            shared: Arc::new(Mutex::new(AudioShared {
                frame: AudioFrame::default(),
                ok: false,
                error: String::new(),
                source: String::new(),
                sens_db: 0.0,
            })),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, AudioShared> {
        lock(&self.shared)
    }

    pub fn start(&mut self) {
        if let Some(t) = &self.thread {
            if !t.is_finished() {
                return;
            }
        }
        self.stop.store(false, Ordering::Relaxed);
        {
            let mut g = lock(&self.shared);
            g.ok = false;
            g.error.clear();
        }
        let shared = self.shared.clone();
        let stop = self.stop.clone();
        self.thread = Some(thread::spawn(move || run(shared, stop)));
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ------------------------------------------------------------------------- //
// Analysis
// ------------------------------------------------------------------------- //
struct Analyzer {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    edges: [usize; BANDS + 1],
    buf: Vec<Complex<f32>>,
    spec: Vec<f32>,
    peak: f32,
    floor: f32,
    bmax: [f32; BANDS],
    onhist: VecDeque<f32>,
    spec_prev: Option<[f32; 9]>,
    bass_prev: f32,
    last_beat: f32,
    lvl_long: f32,
    prev_intimacy: f32,
    started: Instant,
    frame: AudioFrame,
}

impl Analyzer {
    fn new() -> Self {
        let mut edges = [1usize; BANDS + 1];
        for i in 1..=BANDS {
            let v = 240f64.powf(i as f64 / BANDS as f64).round() as usize;
            edges[i] = v.max(edges[i - 1] + 1);
        }
        // np.hanning(N): 0.5 - 0.5 cos(2*pi*k/(N-1))
        let window = (0..BLOCK)
            .map(|k| 0.5 - 0.5 * (std::f32::consts::TAU * k as f32 / (BLOCK - 1) as f32).cos())
            .collect();
        Analyzer {
            fft: FftPlanner::<f32>::new().plan_fft_forward(BLOCK),
            window,
            edges,
            buf: vec![Complex::new(0.0, 0.0); BLOCK],
            spec: vec![0.0; BLOCK / 2 + 1],
            peak: -60.0,
            floor: -60.0,
            bmax: [1e-6; BANDS],
            onhist: VecDeque::with_capacity(41),
            spec_prev: None,
            bass_prev: 0.0,
            last_beat: 0.0,
            lvl_long: 0.0,
            prev_intimacy: 0.0,
            started: Instant::now(),
            frame: AudioFrame::default(),
        }
    }

    fn process(&mut self, x: &[f32], sr: f32, sens_db: f32) {
        let n = BLOCK as f32;
        let dt = n / sr;
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        let db = 20.0 * (rms + 1e-9).log10();

        // adaptive loudness: tracks floor & peak across any OS volume level
        if db > self.peak {
            self.peak = db;
        } else {
            self.peak = (self.peak - 5.0 * dt).max(db);
        }
        if db < self.floor {
            self.floor = db;
        } else {
            self.floor = (self.floor + 3.0 * dt).min(self.peak - 6.0);
        }
        self.floor = self.floor.max(-85.0);
        let span = (self.peak - self.floor).max(8.0);
        let norm = if db < self.floor + 1.2 { 0.0 } else { ((db - self.floor) / span).max(0.0) };
        let gain = 10f32.powf(sens_db / 20.0);
        let level = (norm * gain).powf(1.05).clamp(0.0, 1.0);
        self.frame.level = level;

        // spectrum
        for k in 0..BLOCK {
            self.buf[k] = Complex::new(x[k] * self.window[k], 0.0);
        }
        self.fft.process(&mut self.buf);
        for k in 0..=BLOCK / 2 {
            self.spec[k] = self.buf[k].norm() / (n / 4.0);
        }

        let mut band_norm = [0f32; BANDS];
        for i in 0..BANDS {
            let (e0, e1) = (self.edges[i], self.edges[i + 1]);
            let b = self.spec[e0..e1].iter().sum::<f32>() / (e1 - e0) as f32;
            self.bmax[i] = (self.bmax[i] * 0.985).max(b + 1e-6);
            band_norm[i] = (b / self.bmax[i]).clamp(0.0, 1.0);
            self.frame.bands[i] = (band_norm[i].powf(1.05) * (0.30 + 0.70 * level)).clamp(0.0, 1.0);
        }

        // beat / transient detection: low-frequency spectral flux + bass onset
        let mut spec_low = [0f32; 9];
        spec_low.copy_from_slice(&self.spec[1..10]);
        let prev = self.spec_prev.unwrap_or(spec_low);
        let flux: f32 = spec_low.iter().zip(prev.iter()).map(|(c, p)| (c - p).max(0.0)).sum();
        self.spec_prev = Some(spec_low);

        let bass_e = self.spec[1..4].iter().sum::<f32>() / 3.0;
        let bass_diff = (bass_e - self.bass_prev).max(0.0);
        self.bass_prev = bass_e;

        let onset = flux * 1.8 + bass_diff * 3.0;
        let len = self.onhist.len();
        let avg = if len > 0 { self.onhist.iter().sum::<f32>() / len as f32 } else { 0.0 };
        let std = if len > 5 {
            (self.onhist.iter().map(|v| (v - avg) * (v - avg)).sum::<f32>() / len as f32).sqrt()
        } else {
            0.0
        };
        let thresh = avg + (1.1 * std).max(avg * 0.28) + 1e-5;

        let now = self.started.elapsed().as_secs_f32();
        if len >= 6 && now - self.last_beat > 0.15 && onset > thresh && db > self.floor + 2.0 {
            self.frame.beat_str = ((onset - avg) / (std * 2.0 + 1e-6)).clamp(0.5, 1.0);
            self.frame.beat_id += 1;
            self.last_beat = now;
        }
        self.onhist.push_back(onset);
        if self.onhist.len() > 40 {
            self.onhist.pop_front();
        }

        // musical intelligence: density, intimacy, tension, drop
        let active = band_norm.iter().filter(|&&v| v > 0.20).count() as f32 / BANDS as f32;
        let f = &mut self.frame;
        f.density += (active - f.density) * (1.0 - (-dt / 0.30).exp());

        let mid = band_norm[3..9].iter().sum::<f32>() / 6.0;
        let sub = band_norm[0..3].iter().sum::<f32>() / 3.0;
        let raw_intimacy = (mid * 2.0 * (1.0 - sub * 1.3) * (1.0 - f.density * 0.7)).clamp(0.0, 1.0);
        f.intimacy += (raw_intimacy - f.intimacy) * (1.0 - (-dt / 0.20).exp());

        self.lvl_long += (level - self.lvl_long) * (1.0 - (-dt / 2.0).exp());
        let e_slope = ((level - self.lvl_long) / (self.lvl_long + 0.12)).max(0.0);
        let buildup = mid > 0.26 && sub < 0.42 && e_slope > 0.08;
        if buildup {
            f.tension = (f.tension + dt * 0.45 * (1.0 + e_slope)).min(1.0);
        } else {
            f.tension = (f.tension - dt * 0.40).max(0.0);
        }

        if (f.tension > 0.30 || self.prev_intimacy > 0.4)
            && (bass_diff > 0.30 || (sub > 0.60 && f.density > 0.50))
        {
            f.drop_burst = 1.0;
            f.tension = 0.0;
        }
        f.drop_burst *= (-dt / 1.4).exp();
        self.prev_intimacy = f.intimacy;
    }
}

// ------------------------------------------------------------------------- //
// Capture thread
// ------------------------------------------------------------------------- //
#[cfg(not(feature = "audio"))]
fn run(shared: Arc<Mutex<AudioShared>>, _stop: Arc<AtomicBool>) {
    let mut g = lock(&shared);
    g.ok = false;
    g.error = "built without the `audio` feature".into();
}

#[cfg(feature = "audio")]
fn run(shared: Arc<Mutex<AudioShared>>, stop: Arc<AtomicBool>) {
    use cpal::traits::StreamTrait;

    // The cpal Stream is not Send on every platform, so it lives (and dies) on this thread.
    match capture::open(shared.clone()) {
        Ok((stream, source)) => {
            if let Err(e) = stream.play() {
                let mut g = lock(&shared);
                g.ok = false;
                g.error = format!("{e}");
                return;
            }
            {
                let mut g = lock(&shared);
                g.ok = true;
                g.source = source;
                g.error.clear();
            }
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
            }
            drop(stream);
        }
        Err(e) => {
            let mut g = lock(&shared);
            g.ok = false;
            g.error = e;
        }
    }
}

#[cfg(feature = "audio")]
mod capture {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait};
    use cpal::{FromSample, Sample, SampleFormat, SizedSample};

    /// Windows: WASAPI loopback = building an *input* stream on the default *output* device.
    #[cfg(target_os = "windows")]
    fn pick(host: &cpal::Host) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String), String> {
        let dev = host.default_output_device().ok_or("no default output device")?;
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        let name = dev.name().unwrap_or_default();
        Ok((dev, cfg, format!("loopback: {name}")))
    }

    /// Linux / macOS: look for a monitor / loopback input (PulseAudio monitor, BlackHole, ...).
    #[cfg(not(target_os = "windows"))]
    fn pick(host: &cpal::Host) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String), String> {
        let keys = ["blackhole", "loopback", "soundflower", "monitor", "stereo mix", "what u hear"];
        if let Ok(devs) = host.input_devices() {
            for dev in devs {
                if let Ok(name) = dev.name() {
                    let l = name.to_lowercase();
                    if keys.iter().any(|k| l.contains(k)) {
                        if let Ok(cfg) = dev.default_input_config() {
                            return Ok((dev, cfg, format!("device: {name}")));
                        }
                    }
                }
            }
        }
        Err("no loopback/monitor input device found".into())
    }

    pub fn open(shared: Arc<Mutex<AudioShared>>) -> Result<(cpal::Stream, String), String> {
        let host = cpal::default_host();
        let (dev, cfg, label) = pick(&host)?;
        let sr = cfg.sample_rate().0 as f32;
        let ch = cfg.channels() as usize;
        let sc: cpal::StreamConfig = cfg.clone().into();
        let stream = match cfg.sample_format() {
            SampleFormat::F32 => build::<f32>(&dev, &sc, ch, sr, shared),
            SampleFormat::I16 => build::<i16>(&dev, &sc, ch, sr, shared),
            SampleFormat::U16 => build::<u16>(&dev, &sc, ch, sr, shared),
            SampleFormat::I32 => build::<i32>(&dev, &sc, ch, sr, shared),
            other => Err(format!("unsupported sample format {other:?}")),
        }?;
        Ok((stream, label))
    }

    fn build<T>(
        dev: &cpal::Device,
        cfg: &cpal::StreamConfig,
        ch: usize,
        sr: f32,
        shared: Arc<Mutex<AudioShared>>,
    ) -> Result<cpal::Stream, String>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let mut an = Analyzer::new();
        let mut pending: Vec<f32> = Vec::with_capacity(BLOCK * 4);
        let err_shared = shared.clone();
        dev.build_input_stream(
            cfg,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                for frame in data.chunks(ch.max(1)) {
                    let mut s = 0.0f32;
                    for &v in frame {
                        s += f32::from_sample(v);
                    }
                    pending.push(s / frame.len() as f32);
                }
                while pending.len() >= BLOCK {
                    let sens = lock(&shared).sens_db;
                    an.process(&pending[..BLOCK], sr, sens);
                    pending.drain(..BLOCK);
                    lock(&shared).frame = an.frame;
                }
            },
            move |e| {
                let mut g = lock(&err_shared);
                g.ok = false;
                g.error = format!("{e}");
            },
            None,
        )
        .map_err(|e| e.to_string())
    }
}
