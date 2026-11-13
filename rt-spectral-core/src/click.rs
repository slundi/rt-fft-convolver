//! Time-domain click / impulse repair via a running median despiker.
//!
//! Phase 0 moved click handling out of the STFT chain: a click is a
//! *broadband, flat-spectrum* event that a per-bin operator cannot localise.
//! A short odd-length **median filter** detects it instead — the sample is
//! flagged when it deviates from the window median by more than a robust
//! number of median-absolute-deviations (MAD ≈ a Gaussian σ), then replaced by
//! the median (a cheap local interpolation).
//!
//! The filter runs with a fixed delay of `(window − 1) / 2` samples so the
//! flagged sample sits at the centre of its detection window.  That delay is
//! **always** imposed — even when repair is disabled — so toggling click
//! repair never changes the host's plugin-delay compensation; the `enabled`
//! flag gates only the median substitution.  A clean (or unrepaired) signal is
//! passed through **bit-exact**, just delayed.

/// Largest supported median window; keeps the working buffer on the stack so
/// [`ClickRepair::process_block`] never allocates.
pub const MAX_WINDOW: usize = 9;

/// Click-repair controls (Roadmap-04 Phase 2 "Click controls").
#[derive(Clone, Copy, Debug)]
pub struct ClickParams {
    /// Whether the operator runs at all.
    pub enabled: bool,
    /// Detection threshold in robust standard deviations (MAD-scaled).  A
    /// sample is a click when it deviates from the local median by more than
    /// `sensitivity` σ.  **Higher = less sensitive** (fewer detections);
    /// typical `3…8`.
    pub sensitivity: f32,
    /// Maximum length of a repaired run, in samples.  Longer excursions are
    /// left alone (they are signal, not a click).
    pub max_len: usize,
}

impl Default for ClickParams {
    fn default() -> Self {
        Self {
            enabled: false,
            sensitivity: 5.0,
            max_len: 8,
        }
    }
}

/// Running-median click despiker with a fixed group delay.
#[derive(Debug)]
pub struct ClickRepair {
    /// Odd median-window length (`3..=MAX_WINDOW`).
    window: usize,
    /// Group delay, `(window − 1) / 2`.
    delay: usize,
    /// The last `window` input samples, oldest-first.
    buffer: [f32; MAX_WINDOW],
    /// How many samples have entered since construction / reset (for warm-up).
    filled: usize,
    /// Length of the current repaired run (capped by `max_len`).
    run: usize,
}

/// Scale factor turning the median absolute deviation into a Gaussian-σ
/// estimate.
const MAD_TO_SIGMA: f32 = 1.4826;

impl ClickRepair {
    /// Build a repairer with a median window of `window` samples (forced odd
    /// and clamped to `3..=MAX_WINDOW`).
    ///
    /// **Allocates nothing.**
    pub fn new(window: usize) -> Self {
        let window = (window | 1).clamp(3, MAX_WINDOW); // force odd
        Self {
            window,
            delay: (window - 1) / 2,
            buffer: [0.0; MAX_WINDOW],
            filled: 0,
            run: 0,
        }
    }

    /// The fixed latency (samples) this stage adds while engaged.
    #[inline]
    pub fn latency(&self) -> usize {
        self.delay
    }

    /// Clear the delay line and run state.
    #[inline]
    pub fn reset(&mut self) {
        self.buffer = [0.0; MAX_WINDOW];
        self.filled = 0;
        self.run = 0;
    }

    /// Repair clicks in `buf` in place, always delaying the signal by
    /// [`Self::latency`].  When `params.enabled` is `false` the signal is only
    /// delayed (no median substitution), keeping latency constant.
    ///
    /// # Real-time safety
    /// No allocation (the median scratch lives on the stack).
    pub fn process_block(&mut self, buf: &mut [f32], params: &ClickParams) {
        for x in buf.iter_mut() {
            *x = self.process_sample(*x, params);
        }
    }

    /// Push one input sample, return the delayed (and possibly repaired) output.
    fn process_sample(&mut self, x: f32, params: &ClickParams) -> f32 {
        // Slide the window left and append the new sample.
        self.buffer.copy_within(1..self.window, 0);
        self.buffer[self.window - 1] = x;

        if self.filled < self.window {
            self.filled += 1;
            return 0.0; // warm-up: emit the delay-fill silence
        }

        let win = &self.buffer[..self.window];
        let centre = win[self.delay];
        if !params.enabled {
            // Delay-only: emit the centre sample untouched.
            return centre;
        }

        let median = median(win);
        let deviation = (centre - median).abs();
        let scale = MAD_TO_SIGMA * mad(win, median);

        let is_click = scale > 0.0
            && deviation > params.sensitivity.max(0.0) * scale
            && self.run < params.max_len;

        if is_click {
            self.run += 1;
            median // interpolate the outlier with the local median
        } else {
            self.run = 0;
            centre // clean sample: pass through, only delayed
        }
    }
}

/// Median of a small (`<= MAX_WINDOW`) slice via a stack-array sort.
fn median(win: &[f32]) -> f32 {
    let mut scratch = [0.0f32; MAX_WINDOW];
    let n = win.len();
    scratch[..n].copy_from_slice(win);
    scratch[..n].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    scratch[n / 2]
}

/// Median absolute deviation of `win` about `med`.
fn mad(win: &[f32], med: f32) -> f32 {
    let mut scratch = [0.0f32; MAX_WINDOW];
    let n = win.len();
    for (dst, &s) in scratch[..n].iter_mut().zip(win) {
        *dst = (s - med).abs();
    }
    scratch[..n].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    scratch[n / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn on() -> ClickParams {
        ClickParams {
            enabled: true,
            sensitivity: 4.0,
            max_len: 4,
        }
    }

    #[test]
    fn disabled_delays_but_does_not_repair() {
        // Disabled → the delay is still imposed, but a spike passes through
        // (no median substitution).
        let mut c = ClickRepair::new(5);
        let disabled = ClickParams {
            enabled: false,
            ..on()
        };
        let input: Vec<f32> = (0..64).map(|n| if n == 30 { 5.0 } else { 0.1 }).collect();
        let mut buf = input.clone();
        c.process_block(&mut buf, &disabled);
        let d = c.latency();
        // The spike survives, only shifted by the delay.
        assert_eq!(buf[30 + d], 5.0, "spike should pass through when disabled");
        assert_eq!(buf[20 + d], 0.1);
    }

    #[test]
    fn clean_signal_is_only_delayed() {
        let mut c = ClickRepair::new(5);
        let input: Vec<f32> = (0..2048)
            .map(|n| (2.0 * PI * 3.0 * n as f32 / 2048.0).sin())
            .collect();
        let mut buf = input.clone();
        c.process_block(&mut buf, &on());
        let d = c.latency();
        // A non-flagged sample is emitted bit-exact, just delayed by `d`.
        let mut max_err = 0.0f32;
        for n in (d + 10)..(input.len() - 10) {
            max_err = max_err.max((buf[n] - input[n - d]).abs());
        }
        assert!(max_err < 1e-6, "clean signal distorted by {max_err}");
    }

    #[test]
    fn single_sample_click_is_repaired() {
        let mut c = ClickRepair::new(5);
        let mut input: Vec<f32> = (0..2048)
            .map(|n| 0.2 * (2.0 * PI * 2.0 * n as f32 / 2048.0).sin())
            .collect();
        let click_at = 1000;
        input[click_at] = 1.0; // a full-scale spike on a quiet ±0.2 sine
        let mut buf = input.clone();
        c.process_block(&mut buf, &on());
        let d = c.latency();
        // The spike (now at buf[click_at + d]) must be pulled back toward the
        // local baseline, far from its original 1.0.
        let repaired = buf[click_at + d];
        assert!(repaired.abs() < 0.3, "click not repaired: {repaired}");
    }

    #[test]
    fn long_excursion_is_left_alone() {
        // A run longer than max_len must not be gutted (it is signal).
        let mut c = ClickRepair::new(5);
        let mut input = vec![0.0f32; 512];
        for x in input.iter_mut().take(300).skip(100) {
            *x = 1.0; // a 200-sample plateau, far exceeding max_len = 4
        }
        let mut buf = input.clone();
        c.process_block(&mut buf, &on());
        let d = c.latency();
        // Deep inside the plateau the value survives.
        assert!((buf[200 + d] - 1.0).abs() < 1e-6, "plateau eroded");
    }
}
