//! PSOLA pitch shifter ("Mono Voice" mode) — pitch-synchronous overlap-add.
//!
//! # Idea
//!
//! For a single pitched voice of period `P`, PSOLA lays two-period Hann grains
//! centred on **pitch epochs** spaced `P` apart, then re-emits them on a
//! synthesis grid spaced `P / r`.  Packing the same grains more tightly
//! (`r > 1`) raises the fundamental; spreading them (`r < 1`) lowers it — all
//! while keeping the duration fixed, because analysis and synthesis share one
//! real-time axis.  Grains are duplicated or dropped as the two grids drift.
//!
//! # Scope & simplifications
//!
//! - Epochs are placed at the tracked **period** (uniform spacing) rather than
//!   at glottal-closure instants.  This is the cheap, robust variant; it is
//!   excellent on clean monophonic voice / lead lines and is the reason the
//!   mode is labelled "Mono Voice".
//! - When [`PitchTracker`] reports low confidence (a chord or unpitched
//!   material) the shifter **bypasses** to a dry pass-through and raises
//!   [`Psola::polyphony_warning`], honouring the Roadmap-03 constraint that
//!   PSOLA must not mangle polyphonic input.
//!
//! A `weight` accumulator normalises the overlap-add so the output gain is
//! independent of `r` and of epoch jitter.

use rt_fft_convolver::{DenormalGuard, Ring};

use crate::pitch::PitchTracker;
use crate::shifter::{OnePoleSmoother, Shifter, clamp_ratio};

/// PSOLA pitch shifter.
pub struct Psola {
    tracker: PitchTracker,
    /// Longest period the tracker can report; bounds every grain buffer.
    max_period: usize,
    /// How often (in input samples) to re-estimate the period.
    detect_interval: usize,
    samples_since_detect: usize,

    // ── Raw input, absolute-indexed sliding buffer ──
    raw: Vec<f32>,
    raw_len: usize,
    in_base: usize, // absolute index of raw[0]

    // ── Output overlap-add + weight accumulators ──
    out_acc: Vec<f32>,
    weight_acc: Vec<f32>,
    out_len: usize,
    out_base: usize, // absolute index of out_acc[0]

    // ── Output FIFO ──
    ready: Ring<f32>,

    /// Current smoothed pitch period in samples (0 → not yet locked).
    period: f32,
    /// Absolute time of the next synthesis epoch.
    next_epoch: f64,
    locked: bool,
    polyphonic: bool,

    ratio: OnePoleSmoother,
    max_block: usize,
    sample_rate: f32,
}

impl Psola {
    /// Build a PSOLA shifter for `sample_rate`, tracking `[min_freq, max_freq]`
    /// and absorbing host blocks up to `max_block`.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: f32, min_freq: f32, max_freq: f32, max_block: usize) -> Self {
        let tracker = PitchTracker::new(sample_rate, min_freq, max_freq);
        let max_period = (sample_rate / min_freq).ceil() as usize;
        let detect_frame = tracker.min_frame_len();

        // Grains span two periods; buffers hold several grains plus a block.
        let span = 4 * max_period + max_block + 16;
        let raw_capacity = detect_frame + 4 * max_period + max_block + 16;

        Self {
            tracker,
            max_period,
            detect_interval: max_period, // re-estimate roughly once per period
            samples_since_detect: 0,
            raw: vec![0.0; raw_capacity],
            raw_len: 0,
            in_base: 0,
            out_acc: vec![0.0; span],
            weight_acc: vec![0.0; span],
            out_len: 0,
            out_base: 0,
            ready: Ring::with_capacity(span),
            period: 0.0,
            next_epoch: 0.0,
            locked: false,
            polyphonic: false,
            ratio: OnePoleSmoother::new(1.0, 8.0),
            max_block,
            sample_rate,
        }
    }

    /// Whether the most recent analysis flagged polyphonic / unpitched input
    /// (the plugin surfaces this as a "Mono Voice" warning).
    #[inline]
    pub fn polyphony_warning(&self) -> bool {
        self.polyphonic
    }

    /// Current tracked fundamental in Hz, or `None` before lock.
    pub fn tracked_frequency(&self) -> Option<f32> {
        (self.locked && self.period > 0.0).then(|| self.sample_rate / self.period)
    }

    fn push_raw(&mut self, block: &[f32]) {
        debug_assert!(self.raw_len + block.len() <= self.raw.len());
        self.raw[self.raw_len..self.raw_len + block.len()].copy_from_slice(block);
        self.raw_len += block.len();
    }

    /// Read the raw sample at absolute index `abs` (must be buffered).
    #[inline]
    fn raw_at(&self, abs: usize) -> f32 {
        self.raw[abs - self.in_base]
    }

    /// Re-estimate the pitch period from the freshest buffered audio.
    fn detect(&mut self) {
        let frame_len = self.tracker.min_frame_len();
        if self.raw_len < frame_len {
            return;
        }
        let frame = &self.raw[self.raw_len - frame_len..self.raw_len];
        match self.tracker.analyse(frame) {
            Some(estimate) if estimate.confidence >= 0.8 => {
                self.polyphonic = false;
                if !self.locked {
                    self.period = estimate.period;
                    // First epoch one period into the currently buffered audio.
                    self.next_epoch = (self.in_base + self.max_period) as f64;
                    self.locked = true;
                } else {
                    // Smooth period changes to avoid warble on vibrato.
                    self.period += 0.25 * (estimate.period - self.period);
                }
            }
            _ => self.polyphonic = true,
        }
    }

    /// Produce synthesis grains while enough input is buffered ahead of the
    /// current epoch and the output accumulator has room.
    fn synthesise(&mut self, ratio: f32) {
        if !self.locked || self.period < 1.0 {
            return;
        }
        let period = self.period;
        // PSOLA grains span two *pitch periods* (50 % overlap at unison); a
        // grain wider than that cannot reconstruct at the shifted pitch.
        let grain_half = (period.round() as usize).max(1);
        let grain = 2 * grain_half;
        let synth_spacing = (period / ratio) as f64;
        if synth_spacing < 1.0 {
            return;
        }

        loop {
            let epoch = self.next_epoch;
            // Nearest analysis epoch on the period grid.
            let analysis = ((epoch / period as f64).round() * period as f64) as usize;
            let grain_start = analysis.saturating_sub(grain_half);
            let out_start = (epoch as usize).saturating_sub(grain_half);

            // Need raw covering the grain, and output-accumulator room.
            if grain_start < self.in_base || grain_start + grain >= self.in_base + self.raw_len {
                break;
            }
            if out_start < self.out_base {
                // Epoch fell behind what we already emitted; skip forward.
                self.next_epoch += synth_spacing;
                continue;
            }
            let out_offset = out_start - self.out_base;
            if out_offset + grain >= self.out_acc.len() {
                break;
            }

            // Two-period Hann grain (built lazily from the shared window shape).
            for i in 0..grain {
                let w = hann(i, grain);
                self.out_acc[out_offset + i] += self.raw_at(grain_start + i) * w;
                self.weight_acc[out_offset + i] += w;
            }
            self.out_len = self.out_len.max(out_offset + grain);

            self.next_epoch += synth_spacing;
        }

        // Use the largest possible half-width as a conservative flush/trim
        // margin so no still-active grain can touch a finalised sample.
        let margin = self.max_period;
        self.flush_finished(margin);
        self.trim_raw(margin);
    }

    /// Move samples that no future grain can touch (absolute index below
    /// `next_epoch − half`) from the accumulators into the ready FIFO.
    fn flush_finished(&mut self, half: usize) {
        let limit = (self.next_epoch as usize).saturating_sub(half);
        while self.out_base < limit && self.out_len > 0 {
            let weight = self.weight_acc[0];
            let sample = if weight > 1e-6 {
                self.out_acc[0] / weight
            } else {
                0.0
            };
            self.ready.push(sample);
            // Slide both accumulators left by one and clear the freed tail.
            let tail = self.out_acc.len() - 1;
            self.out_acc.copy_within(1.., 0);
            self.weight_acc.copy_within(1.., 0);
            self.out_acc[tail] = 0.0;
            self.weight_acc[tail] = 0.0;
            self.out_base += 1;
            self.out_len -= 1;
        }
    }

    /// Drop raw history no epoch will read again.
    fn trim_raw(&mut self, half: usize) {
        let keep_from = (self.next_epoch as usize)
            .saturating_sub(2 * half + 1)
            .max(self.in_base);
        let drop = keep_from - self.in_base;
        if drop > 0 && drop <= self.raw_len {
            self.raw.copy_within(drop..self.raw_len, 0);
            self.raw_len -= drop;
            self.in_base += drop;
        }
    }

    /// Dry pass-through used while unlocked or on polyphonic input.
    fn bypass(&mut self, input: &[f32]) {
        self.ready.push_slice(input);
    }

    fn pull(&mut self, out: &mut [f32]) {
        let read = self.ready.pop_slice(out);
        out[read..].fill(0.0);
    }
}

impl Shifter for Psola {
    fn latency_samples(&self) -> usize {
        // The tracker needs a full detection frame before the first grain.
        self.tracker.min_frame_len()
    }

    fn reset(&mut self) {
        self.raw_len = 0;
        self.in_base = 0;
        self.out_acc.fill(0.0);
        self.weight_acc.fill(0.0);
        self.out_len = 0;
        self.out_base = 0;
        self.ready.clear();
        self.period = 0.0;
        self.next_epoch = 0.0;
        self.locked = false;
        self.polyphonic = false;
        self.samples_since_detect = 0;
        self.ratio.snap_to(self.ratio.value());
    }

    fn set_ratio(&mut self, ratio: f32) {
        self.ratio.set_target(clamp_ratio(ratio));
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert!(input.len() <= self.max_block);
        let _denormal_guard = DenormalGuard::new();

        let ratio = self.ratio.tick();
        self.push_raw(input);

        self.samples_since_detect += input.len();
        if self.samples_since_detect >= self.detect_interval {
            self.detect();
            self.samples_since_detect = 0;
        }

        if self.locked && !self.polyphonic {
            self.synthesise(ratio);
        } else {
            self.bypass(input);
            // Keep the raw buffer from growing without bound while bypassed.
            let keep = self.tracker.min_frame_len();
            if self.raw_len > keep {
                let drop = self.raw_len - keep;
                self.raw.copy_within(drop..self.raw_len, 0);
                self.raw_len -= drop;
                self.in_base += drop;
            }
        }

        self.pull(output);
    }
}

/// One sample of a length-`n` periodic Hann window.
#[inline]
fn hann(index: usize, n: usize) -> f32 {
    use std::f32::consts::TAU;
    0.5 - 0.5 * (TAU * index as f32 / n as f32).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dominant_frequency_hz, make_sawtooth, make_sine};

    fn run(shifter: &mut Psola, input: &[f32], block: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        let mut buf = vec![0.0f32; block];
        let mut cursor = 0;
        while cursor < input.len() {
            let end = (cursor + block).min(input.len());
            let len = end - cursor;
            shifter.process_block(&input[cursor..end], &mut buf[..len]);
            output.extend_from_slice(&buf[..len]);
            cursor = end;
        }
        output
    }

    #[test]
    fn unison_preserves_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sawtooth(220.0, sample_rate, 96_000);
        let mut psola = Psola::new(sample_rate, 60.0, 1000.0, 512);
        psola.set_ratio(1.0);
        let output = run(&mut psola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 220.0).abs() < 12.0, "unison freq = {freq}");
    }

    #[test]
    fn octave_up_doubles_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sawtooth(220.0, sample_rate, 96_000);
        let mut psola = Psola::new(sample_rate, 60.0, 1000.0, 512);
        psola.set_ratio(2.0);
        let output = run(&mut psola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 440.0).abs() < 25.0, "octave-up freq = {freq}");
    }

    #[test]
    fn fifth_up_shifts_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sawtooth(220.0, sample_rate, 96_000);
        let mut psola = Psola::new(sample_rate, 60.0, 1000.0, 512);
        psola.set_ratio(1.5); // perfect fifth
        let output = run(&mut psola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 330.0).abs() < 20.0, "fifth freq = {freq}");
    }

    #[test]
    fn polyphonic_input_bypasses() {
        let sample_rate = 48_000.0;
        let a = make_sine(220.0, sample_rate, 48_000);
        let cs = make_sine(277.18, sample_rate, 48_000);
        let e = make_sine(329.63, sample_rate, 48_000);
        let chord: Vec<f32> = (0..a.len()).map(|n| (a[n] + cs[n] + e[n]) / 3.0).collect();
        let mut psola = Psola::new(sample_rate, 60.0, 1000.0, 512);
        psola.set_ratio(2.0);
        let output = run(&mut psola, &chord, 256);
        assert!(psola.polyphony_warning(), "should warn on a chord");
        // Bypassed → dry pass-through: output equals input (unshifted) once the
        // FIFO has filled, so the input energy is faithfully reproduced.
        let tail = &output[output.len() - 8192..];
        let in_tail = &chord[chord.len() - 8192..];
        let out_rms = (tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
        let in_rms = (in_tail.iter().map(|s| s * s).sum::<f32>() / in_tail.len() as f32).sqrt();
        assert!(
            (out_rms - in_rms).abs() < 0.05,
            "bypass rms {out_rms} vs {in_rms}"
        );
    }

    #[test]
    fn silence_stays_silent() {
        let mut psola = Psola::new(48_000.0, 60.0, 1000.0, 256);
        psola.set_ratio(1.5);
        let input = vec![0.0f32; 16_384];
        let output = run(&mut psola, &input, 128);
        let peak = output.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(peak < 1e-6, "silence peak = {peak}");
    }
}
