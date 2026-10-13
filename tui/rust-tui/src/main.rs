//! Music-reactive "planet" - Rust port of sphere3.py.
//!
//! Raw ANSI output, one write per frame inside synchronized-update markers, only
//! changed rows re-sent, mouse reporting never enabled, adaptive quality.
//!
//! Keys:
//!   q / Esc / Ctrl+C : quit        d : toggle demo / system audio
//!   f : show/hide FPS              t : next theme (or 1-4)
//!   Space : pause / resume spin    Up / Down : reactivity    Left / Right : spin speed
//!   b : toggle brightness (1.0x - 2.0x)
//!
//! `sphere --bench` prints headless frame timings (no terminal needed).

mod audio;
mod planet;

use std::io::{self, Write};
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};

use audio::Audio;
use planet::{demo_audio, AudioFrame, Planet, THEME_NAMES};

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    type Handle = *mut c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(n: u32) -> Handle;
        fn GetConsoleMode(h: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: Handle, mode: u32) -> i32;
    }
    #[link(name = "winmm")]
    extern "system" {
        fn timeBeginPeriod(period: u32) -> u32;
    }

    /// 1 ms timer resolution -> accurate sleeps / steady 60 FPS.
    pub fn time_begin_period() {
        unsafe {
            timeBeginPeriod(1);
        }
    }

    /// Disable QuickEdit and console mouse input. QuickEdit can freeze output while the
    /// mouse is over / clicking the console window. (crossterm restores the original mode
    /// on `disable_raw_mode`.)
    pub fn disable_quickedit() {
        unsafe {
            let h = GetStdHandle(-10i32 as u32); // STD_INPUT_HANDLE
            let mut m = 0u32;
            if GetConsoleMode(h, &mut m) != 0 {
                // clear ENABLE_QUICK_EDIT_MODE (0x40) and ENABLE_MOUSE_INPUT (0x10),
                // set ENABLE_EXTENDED_FLAGS (0x80) so the change takes effect
                SetConsoleMode(h, (m & !0x0040 & !0x0010) | 0x0080);
            }
        }
    }
}

struct Hud {
    text: String,
    until: Instant,
}

impl Hud {
    fn say(&mut self, msg: impl Into<String>, secs: f32) {
        self.text = msg.into();
        self.until = Instant::now() + Duration::from_secs_f32(secs);
    }
}

fn bench() {
    for &(w, h) in &[(100usize, 40usize), (200, 60)] {
        let mut p = Planet::new();
        let mut t = 0.0f32;
        let (mut tot, mut mx) = (0.0f64, 0.0f64);
        let frames = 1200;
        for _ in 0..frames {
            t += 0.05;
            let a = demo_audio(t);
            let s = Instant::now();
            p.step(1.0 / 60.0, &a);
            p.render(w, h, t);
            let e = s.elapsed().as_secs_f64() * 1000.0;
            tot += e;
            mx = mx.max(e);
        }
        println!("{w}x{h}: avg {:.3} ms/frame, max {:.3} ms", tot / frames as f64, mx);
    }
}

fn main() {
    if std::env::args().any(|a| a == "--bench") {
        bench();
        return;
    }
    if let Err(e) = run() {
        eprintln!("error: {e}");
    }
}

fn run() -> io::Result<()> {
    #[cfg(windows)]
    win::time_begin_period();

    let mut out = io::stdout().lock();
    // Executing a crossterm command also enables ANSI/VT processing on Windows.
    execute!(out, EnterAlternateScreen, Hide)?;
    terminal::enable_raw_mode()?;
    #[cfg(windows)]
    win::disable_quickedit();
    // mouse reporting OFF (in case a previous program left it on), line wrap OFF, clear
    out.write_all(b"\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?7l\x1b[2J")?;
    out.flush()?;

    let result = main_loop(&mut out);

    let _ = out.write_all(b"\x1b[0m\x1b[?7h");
    let _ = execute!(out, Show, LeaveAlternateScreen);
    let _ = terminal::disable_raw_mode();
    result
}

fn main_loop(out: &mut impl Write) -> io::Result<()> {
    let mut planet = Planet::new();
    let mut audio = Audio::new();
    audio.start();

    let mut use_demo = false;
    let (mut announced, mut warned) = (false, false);
    let mut show_fps = false;
    let mut hud = Hud { text: String::new(), until: Instant::now() };

    let start = Instant::now();
    let mut tprev = start;
    let (mut fps, mut frames, mut fps_t0) = (60.0f32, 0u32, Instant::now());
    let mut work_ema = 0.008f32;
    let target = Duration::from_micros(16_667);
    let target_s = target.as_secs_f32();

    let mut prev: Vec<Vec<u8>> = Vec::new();
    let mut prev_size = (0u16, 0u16);
    let mut status_prev = String::new();
    let mut frame: Vec<u8> = Vec::with_capacity(256 * 1024);
    let mut t_next = Instant::now();

    'main: loop {
        let f0 = Instant::now();

        // ---- input (keyboard only; no mouse events are ever requested) ----
        while event::poll(Duration::ZERO)? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Release {
                    continue;
                }
                if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
                    break 'main;
                }
                match k.code {
                    KeyCode::Esc | KeyCode::Char('q') => break 'main,
                    KeyCode::Char('d') => {
                        if use_demo {
                            use_demo = false;
                            announced = false;
                            warned = false;
                            audio.start();
                            hud.say("audio: system", 2.5);
                        } else {
                            use_demo = true;
                            hud.say("audio: demo", 2.5);
                        }
                    }
                    KeyCode::Char('f') => show_fps = !show_fps,
                    KeyCode::Char('t') => planet.theme = (planet.theme + 1) % THEME_NAMES.len(),
                    KeyCode::Char(c @ '1'..='4') => planet.theme = c as usize - '1' as usize,
                    KeyCode::Up | KeyCode::Char('+') | KeyCode::Char('=') => {
                        let mut g = audio.lock();
                        g.sens_db = (g.sens_db + 3.0).min(18.0);
                        let s = g.sens_db;
                        drop(g);
                        hud.say(format!("reactivity gain {s:+.0} dB"), 2.5);
                    }
                    KeyCode::Down | KeyCode::Char('-') => {
                        let mut g = audio.lock();
                        g.sens_db = (g.sens_db - 3.0).max(-12.0);
                        let s = g.sens_db;
                        drop(g);
                        hud.say(format!("reactivity gain {s:+.0} dB"), 2.5);
                    }
                    KeyCode::Left | KeyCode::Char('[') => {
                        planet.spin_scale = (planet.spin_scale - 0.25).max(0.0);
                        hud.say(format!("spin speed {:.2}x", planet.spin_scale), 2.5);
                    }
                    KeyCode::Right | KeyCode::Char(']') => {
                        planet.spin_scale = (planet.spin_scale + 0.25).min(4.0);
                        hud.say(format!("spin speed {:.2}x", planet.spin_scale), 2.5);
                    }
                    KeyCode::Char(' ') => {
                        planet.spin_scale = if planet.spin_scale > 0.0 { 0.0 } else { 1.0 };
                        hud.say(if planet.spin_scale > 0.0 { "spin active" } else { "spin paused" }, 2.5);
                    }
                    KeyCode::Char('b') => {
                        let next_b = match (planet.brightness * 10.0).round() as i32 {
                            10 => 1.35,
                            13 | 14 => 1.65,
                            16 | 17 => 2.0,
                            _ => 1.0,
                        };
                        planet.brightness = next_b;
                        hud.say(format!("brightness {:.2}x", planet.brightness), 2.5);
                    }
                    _ => {}
                }
            }
        }

        // ---- terminal size (last row = status line) ----
        let (cols, rows) = terminal::size()?;
        let w = (cols as usize).max(20);
        let h = (rows as usize).saturating_sub(1).max(8);
        if (cols, rows) != prev_size {
            prev_size = (cols, rows);
            prev = vec![vec![0xFF]; h]; // sentinel: never equals a real row
            status_prev = String::from("\u{1}");
            out.write_all(b"\x1b[2J")?;
        }

        // ---- audio source ----
        let now = Instant::now();
        let t = (now - start).as_secs_f32();
        let dt = (now - tprev).as_secs_f32();
        tprev = now;

        let fa: AudioFrame = if use_demo {
            demo_audio(t)
        } else {
            let g = audio.lock();
            if g.ok && !announced {
                announced = true;
                hud.say(format!("capturing {}", g.source), 2.5);
            }
            if !g.error.is_empty() && !g.ok && !warned {
                warned = true;
                use_demo = true;
                hud.say(format!("no system audio ({}) - demo mode; 'd' retries", g.error), 5.0);
            }
            g.frame
        };

        planet.step(dt, &fa);
        planet.render(w, h, t);

        // ---- compose frame: only rows that changed ----
        frame.clear();
        frame.extend_from_slice(b"\x1b[?2026h"); // begin synchronized update
        for r in 0..h {
            if planet.rows[r] != prev[r] {
                let _ = write!(frame, "\x1b[{};1H\x1b[2K", r + 1);
                frame.extend_from_slice(&planet.rows[r]);
                frame.extend_from_slice(b"\x1b[0m");
                prev[r].clear();
                prev[r].extend_from_slice(&planet.rows[r]);
            }
        }

        let status = if Instant::now() < hud.until {
            hud.text.clone()
        } else if show_fps {
            format!(
                "{:5.1} FPS | quality {:3.0}% | {} | {}",
                fps,
                planet.quality * 100.0,
                if use_demo { "demo" } else { "system" },
                THEME_NAMES[planet.theme]
            )
        } else {
            String::new()
        };
        if status != status_prev {
            let shown: String = status.chars().take((cols as usize).saturating_sub(2)).collect();
            let _ = write!(frame, "\x1b[{};1H\x1b[2K\x1b[38;2;120;130;150m {}\x1b[0m", rows, shown);
            status_prev = status;
        }
        frame.extend_from_slice(b"\x1b[?2026l"); // end synchronized update
        out.write_all(&frame)?;
        out.flush()?;

        // ---- adaptive quality: keep the frame inside its time budget ----
        let work = f0.elapsed().as_secs_f32();
        work_ema += (work - work_ema) * 0.1;
        if work_ema > target_s * 0.85 {
            planet.set_quality(planet.quality - 0.03);
        } else if work_ema < target_s * 0.55 && planet.quality < 1.0 {
            planet.set_quality(planet.quality + 0.01);
        }

        frames += 1;
        let el = fps_t0.elapsed().as_secs_f32();
        if el >= 0.5 {
            fps = frames as f32 / el;
            frames = 0;
            fps_t0 = Instant::now();
        }

        // ---- precise frame pacing: coarse sleep, then yield until the deadline ----
        t_next += target;
        let now = Instant::now();
        if t_next > now {
            let wait = t_next - now;
            if wait > Duration::from_millis(2) {
                std::thread::sleep(wait - Duration::from_millis(1));
            }
            while Instant::now() < t_next {
                std::thread::yield_now();
            }
        } else {
            t_next = now; // fell behind: don't try to catch up
        }
    }

    audio.stop();
    Ok(())
}
