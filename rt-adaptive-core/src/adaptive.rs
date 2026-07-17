//! Fixed-length adaptive FIR filter with LMS / NLMS coefficient updates.
//!
//! This is the Roadmap-09 Phase 0 core AEC primitive: a tap vector `w` of
//! fixed length `N` that models an unknown linear path (the acoustic echo
//! path from a far-end reference to a near-end microphone) and adapts its taps
//! to null the residual.
//!
//! # Per-sample flow
//!
//! The filter is deliberately split into two O(N) steps so the caller owns the
//! error computation (the echo canceller subtracts the estimate from the mic
//! signal, the line canceller from a delayed input, etc.):
//!
//! ```text
//! y = wᵀ·x          predict(input)  → shift `input` in, return the estimate
//! e = d − y         (caller)         residual after cancellation
//! w += μ·e·x/‖x‖²   adapt(e)         steepest-descent tap update
//! ```
//!
//! # LMS vs NLMS (Phase 0 decision)
//!
//! Plain **LMS** uses a fixed step `μ`; its stable range depends on the input
//! power, so a level change can push it into divergence.  **NLMS** normalises
//! the step by the running input energy `‖x‖²`, staying stable across gain for
//! one extra divide — the default for a live tool.  **RLS** was rejected in
//! Phase 0 (`O(N²)` per sample blows the < 1 % CPU budget).
//!
//! # Real-time safety
//!
//! Every method except [`AdaptiveFilter::nlms`] / [`AdaptiveFilter::lms`] is
//! allocation-free, lock-free and panic-free.  The tap and history storage is
//! the inline `[f32; N]` array, so construction allocates nothing either — the
//! constructors are only non-RT by convention (mirroring the other engines).
//! Weights can drift toward denormals in long silences, so wrap the calling
//! `process` loop in a [`rt_fft_convolver::DenormalGuard`].

/// A steepest-descent adaptive FIR filter with `N` taps.
///
/// `history[0]` is the newest input sample; `weights[i]` multiplies
/// `history[i]`.
#[derive(Clone, Debug)]
pub struct AdaptiveFilter<const N: usize> {
    /// Adaptive tap weights (the estimated impulse response).
    weights: [f32; N],
    /// Tap-delay line of the last `N` inputs, newest first.
    history: [f32; N],
    /// Step size (`μ`).  For NLMS the stable range is `0 < μ < 2`.
    mu: f32,
    /// Regularisation added to `‖x‖²` so a silent reference cannot divide by
    /// zero (NLMS only).
    epsilon: f32,
    /// Running `‖x‖²` of the history, maintained incrementally in O(1).
    energy: f32,
    /// `true` for NLMS (energy-normalised step), `false` for plain LMS.
    normalised: bool,
}

impl<const N: usize> AdaptiveFilter<N> {
    /// A **normalised LMS** filter with step `mu` (typically `0.1 ..= 1.0`).
    ///
    /// **Allocates** nothing, but treated as off-RT by convention.
    pub fn nlms(mu: f32) -> Self {
        Self {
            weights: [0.0; N],
            history: [0.0; N],
            mu,
            epsilon: 1e-6,
            energy: 0.0,
            normalised: true,
        }
    }

    /// A **plain LMS** filter with fixed step `mu`.  Cheaper (no divide) but the
    /// stable `mu` depends on input power.
    pub fn lms(mu: f32) -> Self {
        Self {
            normalised: false,
            ..Self::nlms(mu)
        }
    }

    /// Override the NLMS regularisation term (default `1e-6`).
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon.max(0.0);
    }

    /// Change the step size in place (e.g. from a `Convergence Speed` control).
    #[inline]
    pub fn set_step(&mut self, mu: f32) {
        self.mu = mu;
    }

    /// Number of taps `N`.
    #[inline]
    pub fn len(&self) -> usize {
        N
    }

    /// Always `false` — a zero-length adaptive filter is a construction bug the
    /// const-generic makes impossible, but clippy wants the companion method.
    #[inline]
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// The current tap weights (the estimated impulse response).
    #[inline]
    pub fn weights(&self) -> &[f32; N] {
        &self.weights
    }

    /// Clear both the taps and the delay line.
    pub fn reset(&mut self) {
        self.weights = [0.0; N];
        self.history = [0.0; N];
        self.energy = 0.0;
    }

    /// Shift `input` into the delay line and return the filter output
    /// `y = wᵀ·x`.
    ///
    /// Maintains the running input energy incrementally: the sample leaving the
    /// window is dropped and the new one added, so NLMS needs no O(N) energy
    /// re-scan.
    #[inline]
    pub fn predict(&mut self, input: f32) -> f32 {
        // Update ‖x‖² before the shift discards the oldest sample.
        let leaving = self.history[N - 1];
        self.energy = (self.energy + input * input - leaving * leaving).max(0.0);

        // Shift the delay line one step older, then insert the newest sample.
        self.history.copy_within(0..N - 1, 1);
        self.history[0] = input;

        self.weights
            .iter()
            .zip(self.history.iter())
            .map(|(&w, &x)| w * x)
            .sum()
    }

    /// Adapt the taps to reduce `error` (`= desired − predicted`) for the
    /// history currently in the delay line.
    ///
    /// Call once after each [`predict`](Self::predict).  A no-op when adaptation
    /// should freeze (e.g. the double-talk detector fired) — simply skip it.
    #[inline]
    pub fn adapt(&mut self, error: f32) {
        let step = if self.normalised {
            self.mu / (self.energy + self.epsilon)
        } else {
            self.mu
        };
        let gain = step * error;
        for (weight, &sample) in self.weights.iter_mut().zip(self.history.iter()) {
            *weight += gain * sample;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sine, rms};

    /// A known FIR "echo path"; the filter should identify it from white noise.
    const ECHO_PATH: [f32; 4] = [0.6, -0.3, 0.15, 0.05];

    /// Deterministic pseudo-white noise (xorshift) — no `rand` dependency and
    /// no `Math.random`-style non-determinism in the test.
    fn noise(len: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn nlms_identifies_echo_path() {
        // System identification: reference = white noise, desired = echo of it.
        // NLMS should drive the residual to ~zero and the taps to `ECHO_PATH`.
        let mut filter = AdaptiveFilter::<4>::nlms(0.5);
        let reference = noise(20_000);

        let mut residuals = Vec::with_capacity(reference.len());
        let mut history = [0.0f32; 4];
        for &sample in &reference {
            // Ground-truth echo the filter is trying to reproduce.
            history.copy_within(0..3, 1);
            history[0] = sample;
            let desired: f32 = ECHO_PATH
                .iter()
                .zip(history.iter())
                .map(|(h, x)| h * x)
                .sum();

            let estimate = filter.predict(sample);
            let error = desired - estimate;
            filter.adapt(error);
            residuals.push(error);
        }

        // The converged tail is essentially silent.
        assert!(
            rms(&residuals[15_000..]) < 1e-3,
            "did not converge: tail rms {}",
            rms(&residuals[15_000..])
        );
        // …and the taps match the true path.
        for (got, want) in filter.weights().iter().zip(ECHO_PATH.iter()) {
            assert!((got - want).abs() < 1e-2, "tap {got} != {want}");
        }
    }

    #[test]
    fn lms_also_converges_on_stationary_input() {
        // Plain LMS is stable at a small step for a unit-power reference.
        let mut filter = AdaptiveFilter::<4>::lms(0.05);
        let reference = noise(40_000);
        let mut history = [0.0f32; 4];
        let mut last = 0.0;
        for &sample in &reference {
            history.copy_within(0..3, 1);
            history[0] = sample;
            let desired: f32 = ECHO_PATH
                .iter()
                .zip(history.iter())
                .map(|(h, x)| h * x)
                .sum();
            let estimate = filter.predict(sample);
            last = desired - estimate;
            filter.adapt(last);
        }
        assert!(last.abs() < 0.05, "LMS residual too large: {last}");
    }

    #[test]
    fn reset_clears_taps_and_state() {
        let mut filter = AdaptiveFilter::<8>::nlms(0.5);
        for &sample in &noise(1_000) {
            let e = 0.5 - filter.predict(sample);
            filter.adapt(e);
        }
        filter.reset();
        assert!(filter.weights().iter().all(|&w| w == 0.0));
        assert_eq!(filter.predict(0.0), 0.0);
    }

    #[test]
    fn frozen_adaptation_holds_the_taps() {
        // Not calling `adapt` must leave the weights untouched (double-talk).
        let mut filter = AdaptiveFilter::<4>::nlms(0.5);
        let tone = make_sine(1_000.0, 48_000.0, 256);
        for &sample in &tone {
            filter.predict(sample);
            // deliberately no adapt()
        }
        assert!(filter.weights().iter().all(|&w| w == 0.0));
    }
}
