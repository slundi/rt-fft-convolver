//! Framework-agnostic audio-thread processor and off-RT log worker.
//!
//! [`MeterProcessor`] is the resident audio-thread renderer: it feeds the
//! [`LoudnessMeter`], applies `Start/Stop` and edge-triggered `Reset` in place,
//! and publishes every block to a lock-free [`MeterBroadcast`].  Because a
//! meter is analysis-only and nothing reallocates on a parameter change, there
//! is no off-thread controller — the processor is built once in `initialize`.
//!
//! [`MeterLogger`] is the off-RT side: a worker thread that polls the same
//! broadcast at a fixed cadence and appends CSV / JSON rows to a file.  It
//! never touches the audio thread — it only reads the wait-free broadcast.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rt_fft_convolver::DenormalGuard;
use rt_loudness_core::{LoudnessMeter, MeterBroadcast, MeterValues};

use crate::params::LoudnessParams;

/// Upper bound on channels the non-allocating channel-pointer buffer holds.
/// Mono + stereo (v1) fit easily; sized for 7.1 so v2 needs no change here.
const MAX_CHANNELS: usize = 8;

/// The audio-thread meter renderer.
pub struct MeterProcessor {
    meter: LoudnessMeter,
    broadcast: Arc<MeterBroadcast>,
    /// Previous `Reset` value, for rising-edge detection.
    prev_reset: bool,
}

impl MeterProcessor {
    /// Build a processor for `channels` channels at `sample_rate` Hz, sized for
    /// blocks up to `max_block`, publishing to `broadcast`.
    ///
    /// **Allocates** — call off the audio thread (`initialize`).
    pub fn new(
        channels: usize,
        sample_rate: u32,
        max_block: usize,
        broadcast: Arc<MeterBroadcast>,
    ) -> Self {
        Self {
            meter: LoudnessMeter::new(channels, sample_rate, max_block),
            broadcast,
            prev_reset: false,
        }
    }

    /// Build with an explicit true-peak half-band design (lighter designs keep
    /// tests fast).
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_true_peak_design(
        channels: usize,
        sample_rate: u32,
        max_block: usize,
        broadcast: Arc<MeterBroadcast>,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        Self {
            meter: LoudnessMeter::with_true_peak_design(
                channels,
                sample_rate,
                max_block,
                atten_db,
                transition_frac,
            ),
            broadcast,
            prev_reset: false,
        }
    }

    /// The meter is analysis-only: zero latency in the audio path.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        0
    }

    /// A clone of the shared broadcast handle (for the editor / logger).
    pub fn broadcast(&self) -> Arc<MeterBroadcast> {
        self.broadcast.clone()
    }

    /// Analyse one block of every channel and publish the readings.  The audio
    /// is **not** modified.
    ///
    /// **Real-time safe** for blocks `<= max_block`.
    pub fn process(&mut self, channels: &[&[f32]], params: &LoudnessParams) {
        // Flush denormals over the whole callback (near-silent K-weighted tail).
        let _guard = DenormalGuard::new();

        // `Reset` on a rising edge only, so holding the control does not wipe
        // the measurement every block.
        if params.reset && !self.prev_reset {
            self.meter.reset();
        }
        self.prev_reset = params.reset;

        // `Start/Stop` gates integrated / LRA; momentary / short-term / peaks
        // keep updating either way.
        self.meter.set_integrating(params.running);

        self.meter.process_block(channels);
        self.meter.publish(&self.broadcast);
    }

    /// Fill a fixed stack buffer of immutable channel references from a
    /// `nih-plug`-style mutable channel slice and analyse it — no allocation.
    ///
    /// **Real-time safe.**
    pub fn process_buffer(&mut self, channels: &mut [&mut [f32]], params: &LoudnessParams) {
        let mut refs: [&[f32]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
        let count = channels.len().min(MAX_CHANNELS);
        for (slot, channel) in refs.iter_mut().zip(channels.iter()) {
            *slot = &channel[..];
        }
        self.process(&refs[..count], params);
    }

    /// The current meter readings.
    pub fn values(&self) -> MeterValues {
        self.meter.values()
    }

    /// Clear the meter and publish the (silent) result.
    pub fn reset(&mut self) {
        self.meter.reset();
        self.prev_reset = false;
        self.meter.publish(&self.broadcast);
    }
}

/// Output file format for [`MeterLogger`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Comma-separated values with a header row.
    Csv,
    /// One JSON object per line (JSON Lines).
    Json,
}

/// The CSV header row (also documents the column order for JSON).
pub fn csv_header() -> &'static str {
    "elapsed_s,momentary_lufs,short_term_lufs,integrated_lufs,lra_lu,true_peak_dbtp,sample_peak_dbfs,rms_db\n"
}

/// Format one `f32`, leaving non-finite values (silence `−∞`) blank so the log
/// does not carry `-inf` tokens a spreadsheet would choke on.
fn field(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.2}")
    } else {
        String::new()
    }
}

/// Format one JSON field: a number, or `null` for non-finite values.
fn json_field(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.2}")
    } else {
        "null".to_string()
    }
}

/// Format one log row for `format`, at `elapsed_s` seconds into the session.
pub fn format_row(format: LogFormat, elapsed_s: f64, values: &MeterValues) -> String {
    match format {
        LogFormat::Csv => format!(
            "{:.2},{},{},{},{},{},{},{}\n",
            elapsed_s,
            field(values.momentary_lufs),
            field(values.short_term_lufs),
            field(values.integrated_lufs),
            field(values.lra_lu),
            field(values.true_peak_dbtp),
            field(values.sample_peak_dbfs),
            field(values.rms_db),
        ),
        LogFormat::Json => format!(
            "{{\"elapsed_s\":{:.2},\"momentary_lufs\":{},\"short_term_lufs\":{},\
             \"integrated_lufs\":{},\"lra_lu\":{},\"true_peak_dbtp\":{},\
             \"sample_peak_dbfs\":{},\"rms_db\":{}}}\n",
            elapsed_s,
            json_field(values.momentary_lufs),
            json_field(values.short_term_lufs),
            json_field(values.integrated_lufs),
            json_field(values.lra_lu),
            json_field(values.true_peak_dbtp),
            json_field(values.sample_peak_dbfs),
            json_field(values.rms_db),
        ),
    }
}

/// Off-RT worker that appends meter readings to a file at a fixed cadence.
///
/// Owns a thread that polls the lock-free [`MeterBroadcast`] — it never blocks
/// or allocates on the audio thread.  Dropping (or [`MeterLogger::stop`]) joins
/// the worker.
pub struct MeterLogger {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MeterLogger {
    /// Start logging `broadcast` to `path` in `format`, sampling every `period`.
    ///
    /// **Off-RT** — opens the file and spawns a thread; never call from the
    /// audio thread (route through the host's background executor).
    pub fn start(
        path: &Path,
        format: LogFormat,
        broadcast: Arc<MeterBroadcast>,
        period: Duration,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        if format == LogFormat::Csv {
            writer.write_all(csv_header().as_bytes())?;
            writer.flush()?;
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = thread::spawn(move || {
            let start = Instant::now();
            // Write-then-sleep so the first row lands immediately.
            while !stop_thread.load(Ordering::Acquire) {
                let elapsed = start.elapsed().as_secs_f64();
                let row = format_row(format, elapsed, &broadcast.load());
                if writer.write_all(row.as_bytes()).is_err() {
                    break;
                }
                let _ = writer.flush();
                thread::sleep(period);
            }
        });

        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }

    /// Signal the worker to stop and wait for it to finish.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for MeterLogger {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const SR: u32 = 48_000;
    const LIGHT: (f64, f64) = (80.0, 0.1);

    fn processor(channels: usize, max_block: usize) -> MeterProcessor {
        MeterProcessor::with_true_peak_design(
            channels,
            SR,
            max_block,
            Arc::new(MeterBroadcast::new()),
            LIGHT.0,
            LIGHT.1,
        )
    }

    fn sine(freq: f32, samples: usize, amp: f32) -> Vec<f32> {
        (0..samples)
            .map(|index| amp * (TAU * freq * index as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn process_publishes_to_broadcast() {
        let mut proc = processor(2, 4_800);
        let broadcast = proc.broadcast();
        let signal = sine(1_000.0, 4_800, 0.5);
        let params = LoudnessParams::default();
        for _ in 0..10 {
            proc.process(&[&signal, &signal], &params);
        }
        assert_eq!(broadcast.load(), proc.values());
        assert!(proc.values().momentary_lufs.is_finite());
    }

    #[test]
    fn reset_edge_clears_measurement() {
        let mut proc = processor(2, 4_800);
        let signal = sine(1_000.0, 4_800, 0.5);
        let mut params = LoudnessParams::default();
        for _ in 0..10 {
            proc.process(&[&signal, &signal], &params);
        }
        assert!(proc.values().integrated_lufs.is_finite());

        // Rising edge on `reset`.
        params.reset = true;
        let silence = vec![0.0f32; 4_800];
        proc.process(&[&silence, &silence], &params);
        assert_eq!(proc.values().integrated_lufs, f32::NEG_INFINITY);
    }

    #[test]
    fn stop_freezes_integrated() {
        let mut proc = processor(2, 4_800);
        let quiet = sine(1_000.0, 4_800, 0.1);
        let mut params = LoudnessParams::default();
        for _ in 0..10 {
            proc.process(&[&quiet, &quiet], &params);
        }
        let frozen = proc.values().integrated_lufs;

        // Stop, then blast a loud tone: integrated must not move.
        params.running = false;
        let loud = sine(1_000.0, 4_800, 0.9);
        for _ in 0..10 {
            proc.process(&[&loud, &loud], &params);
        }
        assert!((proc.values().integrated_lufs - frozen).abs() < 0.1);
        // Momentary tracked the loud tone.
        assert!(proc.values().momentary_lufs > frozen);
    }

    #[test]
    fn process_buffer_matches_process() {
        let signal = sine(1_000.0, 512, 0.5);
        let params = LoudnessParams::default();

        let mut via_slices = processor(2, 512);
        via_slices.process(&[&signal, &signal], &params);

        let mut via_buffer = processor(2, 512);
        let mut left = signal.clone();
        let mut right = signal.clone();
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        via_buffer.process_buffer(&mut channels, &params);

        assert_eq!(via_buffer.values(), via_slices.values());
    }

    #[test]
    fn latency_is_zero() {
        assert_eq!(processor(2, 512).latency_samples(), 0);
    }

    #[test]
    fn csv_row_blanks_non_finite() {
        let mut values = MeterValues::SILENCE;
        values.momentary_lufs = -14.23;
        let row = format_row(LogFormat::Csv, 1.5, &values);
        // elapsed, momentary present; the rest (still −∞) blank.
        assert!(row.starts_with("1.50,-14.23,,"));
        assert!(row.ends_with('\n'));
    }

    #[test]
    fn json_row_nulls_non_finite() {
        let mut values = MeterValues::SILENCE;
        values.integrated_lufs = -23.0;
        let row = format_row(LogFormat::Json, 2.0, &values);
        assert!(row.contains("\"integrated_lufs\":-23.00"));
        assert!(row.contains("\"momentary_lufs\":null"));
        assert!(row.trim_end().ends_with('}'));
    }

    #[test]
    fn logger_writes_rows_to_file() {
        use std::io::Read;

        let dir = std::env::temp_dir();
        let path = dir.join(format!("rt-loudness-log-{}.csv", std::process::id()));

        let broadcast = Arc::new(MeterBroadcast::new());
        let mut values = MeterValues::SILENCE;
        values.momentary_lufs = -20.0;
        broadcast.store(&values);

        let logger = MeterLogger::start(
            &path,
            LogFormat::Csv,
            broadcast.clone(),
            Duration::from_millis(5),
        )
        .expect("start logger");
        thread::sleep(Duration::from_millis(40));
        logger.stop();

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(contents.starts_with("elapsed_s,"), "missing CSV header");
        let data_rows = contents.lines().skip(1).count();
        assert!(data_rows >= 1, "expected at least one data row");
        assert!(
            contents.contains("-20.00"),
            "logged momentary value missing"
        );
    }
}
