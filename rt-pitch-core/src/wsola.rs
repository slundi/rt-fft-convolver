//! WSOLA pitch shifter ("Tight/Mono" mode) — time domain, no FFT.
//!
//! # Pitch shift = resample ∘ time-stretch
//!
//! A time-domain pitch shift is a **resample** (which changes pitch *and*
//! duration together) followed by a **time-stretch** that restores the
//! original duration.  To transpose up by ratio `r` we first read the input
//! `r`× faster (linear interpolation) — shortening it by `r` and raising its
//! pitch by `r` — then stretch it back by `r` so the output length matches the
//! input again.  The pitch change survives; the duration change cancels.
//!
//! # Why WSOLA for the stretch
//!
//! Naïve overlap-add time-stretch repositions grains on a fixed synthesis grid
//! and creates phase discontinuities that "warble".  **WSOLA** (Waveform
//! Similarity Overlap-Add) instead searches a small tolerance window for the
//! grain whose waveform best continues the previously emitted grain
//! (max cross-correlation with the "natural progression"), so successive
//! grains splice smoothly.  It is cheap and keeps transients crisp, but the
//! similarity search assumes a single dominant period — hence the "Mono"
//! label: it warbles on chords.

use rt_fft_convolver::{DenormalGuard, Ring};

use crate::shifter::{OnePoleSmoother, Shifter, clamp_ratio};
use crate::window::{Window, WindowKind};

/// WSOLA pitch shifter.
pub struct Wsola {
    /// Grain / window length `L`.
    frame: usize,
    /// Synthesis hop `Hs` (fixed, `L/2` for 50 % overlap).
    syn_hop: usize,
    /// Similarity-search radius in samples.
    tolerance: usize,
    /// Hann window applied to every grain.
    window: Vec<f32>,

    // ── Stage 1: resampler (input → pitched, wrong-duration) ──
    /// Sliding buffer of raw input awaiting resampling.
    raw: Vec<f32>,
    raw_len: usize,
    /// Fractional read position into `raw`.
    resample_pos: f64,

    // ── Stage 2: WSOLA time-stretch (pitched → output) ──
    /// Sliding buffer of resampled ("pitched") samples awaiting stretching.
    pitched: Vec<f32>,
    pitched_len: usize,
    /// Fractional analysis position into `pitched`.
    analysis_pos: f64,
    /// "Natural progression" reference the next grain is matched against.
    reference: Vec<f32>,
    have_reference: bool,
    /// Overlap-add accumulator for the stretched output (length `L`).
    stretch_acc: Vec<f32>,

    // ── Output FIFO ──
    ready: Ring<f32>,

    ratio: OnePoleSmoother,
    max_block: usize,
}

impl Wsola {
    /// Build a WSOLA shifter with grain length `frame` (even), absorbing host
    /// blocks up to `max_block`.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(frame: usize, max_block: usize) -> Self {
        assert!(
            frame >= 4 && frame.is_multiple_of(2),
            "frame must be even ≥ 4"
        );
        let syn_hop = frame / 2;
        let tolerance = frame / 4;
        let window = Window::new(WindowKind::Hann, frame);

        // Buffer sizes are generous upper bounds so the hot path never grows a
        // Vec.  At the max ratio the resampler consumes up to `MAX_RATIO ×`
        // input per output sample; the pitched buffer must hold a grain plus
        // the search tolerance plus a block's worth of headroom.
        let raw_capacity = frame + max_block * 8 + 16;
        let pitched_capacity = frame * 4 + tolerance * 2 + max_block * 4 + 16;
        let ready_capacity = frame * 4 + max_block * 4 + 16;

        Self {
            frame,
            syn_hop,
            tolerance,
            window: window.as_slice().to_vec(),
            raw: vec![0.0; raw_capacity],
            raw_len: 0,
            resample_pos: 0.0,
            pitched: vec![0.0; pitched_capacity],
            pitched_len: 0,
            analysis_pos: 0.0,
            reference: vec![0.0; frame],
            have_reference: false,
            stretch_acc: vec![0.0; frame],
            ready: Ring::with_capacity(ready_capacity),
            ratio: OnePoleSmoother::new(1.0, 8.0),
            max_block,
        }
    }

    /// Append a host block to the raw input buffer.
    fn push_raw(&mut self, block: &[f32]) {
        debug_assert!(self.raw_len + block.len() <= self.raw.len());
        self.raw[self.raw_len..self.raw_len + block.len()].copy_from_slice(block);
        self.raw_len += block.len();
    }

    /// Drop the first `count` samples of a sliding buffer, sliding the rest to
    /// the front.  Returns the new fill length.
    fn slide(buffer: &mut [f32], len: usize, count: usize) -> usize {
        if count == 0 {
            return len;
        }
        let count = count.min(len);
        buffer.copy_within(count..len, 0);
        len - count
    }

    /// Stage 1 — resample `raw` at the current ratio into `pitched`.
    fn resample(&mut self, ratio: f32) {
        let step = ratio as f64;
        // Produce pitched samples while a full interpolation stencil is
        // available and the pitched buffer has room.
        while (self.resample_pos as usize) + 1 < self.raw_len
            && self.pitched_len < self.pitched.len()
        {
            let index = self.resample_pos as usize;
            let frac = (self.resample_pos - index as f64) as f32;
            let sample = self.raw[index] * (1.0 - frac) + self.raw[index + 1] * frac;
            self.pitched[self.pitched_len] = sample;
            self.pitched_len += 1;
            self.resample_pos += step;
        }
        // Discard raw samples now fully behind the read position.
        let consumed = self.resample_pos as usize;
        if consumed > 0 {
            self.raw_len = Self::slide(&mut self.raw, self.raw_len, consumed);
            self.resample_pos -= consumed as f64;
        }
    }

    /// Stage 2 — WSOLA time-stretch `pitched` by `ratio` into `ready`.
    fn stretch(&mut self, ratio: f32) {
        // Analysis hop that yields a stretch factor of `ratio`
        // (output/input length = Hs / Ha = ratio).
        let analysis_hop = self.syn_hop as f64 / ratio as f64;
        // Samples that must exist ahead of the analysis position: the grain,
        // the search tolerance, and the extra synthesis hop the natural-
        // progression reference reaches into.
        let need = self.frame + self.tolerance + self.syn_hop;

        while (self.analysis_pos as usize) + need < self.pitched_len
            && self.ready.free() >= self.syn_hop
        {
            let base = self.analysis_pos as usize;

            // Similarity search: pick the grain offset best matching the
            // natural progression of the previous grain.
            let offset = if self.have_reference {
                self.best_offset(base)
            } else {
                0
            };
            let start = (base as isize + offset).max(0) as usize;

            // Window the grain and overlap-add into the accumulator.
            for i in 0..self.frame {
                self.stretch_acc[i] += self.pitched[start + i] * self.window[i];
            }
            // Emit one synthesis hop.
            self.ready.push_slice(&self.stretch_acc[..self.syn_hop]);
            // Slide the accumulator left by a synthesis hop.
            self.stretch_acc.copy_within(self.syn_hop.., 0);
            self.stretch_acc[self.frame - self.syn_hop..].fill(0.0);

            // The natural progression: what follows this grain by one synthesis
            // hop — the next grain should continue it smoothly.
            let ref_start = start + self.syn_hop;
            self.reference
                .copy_from_slice(&self.pitched[ref_start..ref_start + self.frame]);
            self.have_reference = true;

            self.analysis_pos += analysis_hop;
        }

        // Trim consumed pitched samples, keeping enough history for the search.
        let keep_back = self.tolerance + 1;
        let base = self.analysis_pos as usize;
        if base > keep_back {
            let drop = base - keep_back;
            self.pitched_len = Self::slide(&mut self.pitched, self.pitched_len, drop);
            self.analysis_pos -= drop as f64;
        }
    }

    /// Cross-correlate candidate grains in `[base − δ, base + δ]` against the
    /// stored reference; return the best offset (relative to `base`).
    fn best_offset(&self, base: usize) -> isize {
        let delta = self.tolerance as isize;
        let mut best_offset = 0isize;
        let mut best_score = f32::NEG_INFINITY;
        for candidate in -delta..=delta {
            let start = base as isize + candidate;
            if start < 0 || start as usize + self.frame > self.pitched_len {
                continue;
            }
            let start = start as usize;
            let mut score = 0.0f32;
            for i in 0..self.frame {
                score += self.pitched[start + i] * self.reference[i];
            }
            if score > best_score {
                best_score = score;
                best_offset = candidate;
            }
        }
        best_offset
    }

    /// Copy up to `out.len()` finished output samples, zero-padding shortfall.
    fn pull(&mut self, out: &mut [f32]) {
        let read = self.ready.pop_slice(out);
        out[read..].fill(0.0);
    }
}

impl Shifter for Wsola {
    fn latency_samples(&self) -> usize {
        // One grain of buffering dominates the group delay.
        self.frame
    }

    fn reset(&mut self) {
        self.raw_len = 0;
        self.resample_pos = 0.0;
        self.pitched_len = 0;
        self.analysis_pos = 0.0;
        self.have_reference = false;
        self.reference.fill(0.0);
        self.stretch_acc.fill(0.0);
        self.ready.clear();
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
        self.resample(ratio);
        self.stretch(ratio);
        self.pull(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dominant_frequency_hz, make_sine};

    fn run(shifter: &mut Wsola, input: &[f32], block: usize) -> Vec<f32> {
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
        let input = make_sine(440.0, sample_rate, 48_000);
        let mut wsola = Wsola::new(1024, 512);
        wsola.set_ratio(1.0);
        let output = run(&mut wsola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 440.0).abs() < 20.0, "unison freq = {freq}");
    }

    #[test]
    fn octave_up_doubles_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 48_000);
        let mut wsola = Wsola::new(1024, 512);
        wsola.set_ratio(2.0);
        let output = run(&mut wsola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 880.0).abs() < 25.0, "octave-up freq = {freq}");
    }

    #[test]
    fn octave_down_halves_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 48_000);
        let mut wsola = Wsola::new(1024, 512);
        wsola.set_ratio(0.5);
        let output = run(&mut wsola, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 220.0).abs() < 15.0, "octave-down freq = {freq}");
    }

    #[test]
    fn silence_stays_silent() {
        let mut wsola = Wsola::new(512, 256);
        wsola.set_ratio(1.5);
        let input = vec![0.0f32; 8192];
        let output = run(&mut wsola, &input, 128);
        let peak = output.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(peak < 1e-6, "silence peak = {peak}");
    }
}
