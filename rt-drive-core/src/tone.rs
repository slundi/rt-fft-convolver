//! Biquad-based tone stacks.
//!
//! The roadmap models the tone stack per pedal (not IR-based).  This module
//! provides:
//!
//! - [`Biquad`] — a single RBJ-cookbook second-order section (the building
//!   block), with shelving and peaking designers.
//! - [`Baxandall`] — the default 2-band (bass/treble) shelving tone stack.
//! - [`ThreeBand`] — a bass/mid/treble stack (low-shelf + peaking mid +
//!   high-shelf) approximating a Fender/Marshall-style 3-band control.
//!
//! Coefficients are designed once in the `new()` / `set_*` methods (which use
//! transcendentals and are **not** real-time safe); the per-sample `process`
//! is a branch-free biquad and **is** real-time safe.
//!
//! Reference: Robert Bristow-Johnson, *Audio EQ Cookbook*.

use std::f32::consts::TAU;

/// A single second-order IIR section (biquad), Transposed Direct-Form II.
///
/// State (`z1`, `z2`) is carried between calls; coefficients are normalised by
/// `a0` at design time so `process` needs no division.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// An identity (unity-gain, all-pass-at-DC) section: `y = x`.
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Low-shelf at `freq_hz` with `gain_db` of boost/cut and shelf slope
    /// `S = 1` (the maximally-flat shelf).
    pub fn low_shelf(freq_hz: f32, sample_rate: f32, gain_db: f32) -> Self {
        let (amp, cos_w0, two_sqrt_a_alpha) = shelf_terms(freq_hz, sample_rate, gain_db);
        let b0 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w0);
        let b2 = amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w0);
        let a2 = (amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
        Self::normalised(b0, b1, b2, a0, a1, a2)
    }

    /// High-shelf at `freq_hz` with `gain_db` and shelf slope `S = 1`.
    pub fn high_shelf(freq_hz: f32, sample_rate: f32, gain_db: f32) -> Self {
        let (amp, cos_w0, two_sqrt_a_alpha) = shelf_terms(freq_hz, sample_rate, gain_db);
        let b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w0);
        let b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a0 = (amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w0);
        let a2 = (amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
        Self::normalised(b0, b1, b2, a0, a1, a2)
    }

    /// Peaking (bell) EQ at `freq_hz`, `gain_db`, quality `q`.
    pub fn peaking(freq_hz: f32, sample_rate: f32, gain_db: f32, q: f32) -> Self {
        let amp = 10.0f32.powf(gain_db / 40.0);
        let w0 = TAU * (freq_hz / sample_rate);
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q.max(1e-4));
        let b0 = 1.0 + alpha * amp;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * amp;
        let a0 = 1.0 + alpha / amp;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / amp;
        Self::normalised(b0, b1, b2, a0, a1, a2)
    }

    /// Build from raw coefficients, normalising by `a0`.
    fn normalised(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        let inv_a0 = 1.0 / a0;
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Filter one sample (Transposed Direct-Form II).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    /// Reset the filter memory.
    #[inline]
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Magnitude of the frequency response at normalised frequency
    /// `freq_frac = f / fs ∈ [0, 0.5]`.  Used by tests to verify shelf gains.
    #[cfg(test)]
    fn magnitude(&self, freq_frac: f32) -> f32 {
        // Evaluate H(e^{jω}) = (b0 + b1 z⁻¹ + b2 z⁻²)/(1 + a1 z⁻¹ + a2 z⁻²).
        let omega = TAU * freq_frac;
        let (sin1, cos1) = omega.sin_cos();
        let (sin2, cos2) = (2.0 * omega).sin_cos();
        let num_re = self.b0 + self.b1 * cos1 + self.b2 * cos2;
        let num_im = -(self.b1 * sin1 + self.b2 * sin2);
        let den_re = 1.0 + self.a1 * cos1 + self.a2 * cos2;
        let den_im = -(self.a1 * sin1 + self.a2 * sin2);
        let num_mag = (num_re * num_re + num_im * num_im).sqrt();
        let den_mag = (den_re * den_re + den_im * den_im).sqrt();
        num_mag / den_mag
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self::identity()
    }
}

/// Shared shelf coefficient terms: `(A, cos ω₀, 2√A·α)` for slope `S = 1`.
fn shelf_terms(freq_hz: f32, sample_rate: f32, gain_db: f32) -> (f32, f32, f32) {
    let amp = 10.0f32.powf(gain_db / 40.0);
    let w0 = TAU * (freq_hz / sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    // Shelf slope S = 1: α = (sin ω₀ / 2) · √((A + 1/A)(1/S − 1) + 2); with
    // S = 1 the (1/S − 1) term vanishes → α = (sin ω₀ / 2)·√2.
    let alpha = (sin_w0 / 2.0) * 2.0f32.sqrt();
    let two_sqrt_a_alpha = 2.0 * amp.sqrt() * alpha;
    (amp, cos_w0, two_sqrt_a_alpha)
}

/// The default 2-band Baxandall tone stack: a low shelf and a high shelf in
/// series.
#[derive(Debug, Clone, Copy)]
pub struct Baxandall {
    low: Biquad,
    high: Biquad,
}

impl Baxandall {
    /// Default corner frequencies for the two shelves (Hz).
    pub const BASS_HZ: f32 = 120.0;
    /// Treble shelf corner (Hz).
    pub const TREBLE_HZ: f32 = 3_000.0;

    /// Create a Baxandall stack with the default corners and the given
    /// bass / treble gains in dB.
    pub fn new(sample_rate: f32, bass_db: f32, treble_db: f32) -> Self {
        Self {
            low: Biquad::low_shelf(Self::BASS_HZ, sample_rate, bass_db),
            high: Biquad::high_shelf(Self::TREBLE_HZ, sample_rate, treble_db),
        }
    }

    /// Filter one sample through both shelves.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.high.process_sample(self.low.process_sample(input))
    }

    /// Filter a block in place. **Real-time safe.**
    #[inline]
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset both shelves.
    #[inline]
    pub fn reset(&mut self) {
        self.low.reset();
        self.high.reset();
    }
}

/// A 3-band tone stack: low shelf + peaking mid + high shelf.
#[derive(Debug, Clone, Copy)]
pub struct ThreeBand {
    low: Biquad,
    mid: Biquad,
    high: Biquad,
}

impl ThreeBand {
    /// Default bass shelf corner (Hz).
    pub const BASS_HZ: f32 = 100.0;
    /// Default mid peak centre (Hz).
    pub const MID_HZ: f32 = 750.0;
    /// Default mid peak quality.
    pub const MID_Q: f32 = 0.7;
    /// Default treble shelf corner (Hz).
    pub const TREBLE_HZ: f32 = 3_000.0;

    /// Create a 3-band stack with default corners and the given band gains.
    pub fn new(sample_rate: f32, bass_db: f32, mid_db: f32, treble_db: f32) -> Self {
        Self {
            low: Biquad::low_shelf(Self::BASS_HZ, sample_rate, bass_db),
            mid: Biquad::peaking(Self::MID_HZ, sample_rate, mid_db, Self::MID_Q),
            high: Biquad::high_shelf(Self::TREBLE_HZ, sample_rate, treble_db),
        }
    }

    /// Filter one sample through all three bands. **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.high
            .process_sample(self.mid.process_sample(self.low.process_sample(input)))
    }

    /// Filter a block in place. **Real-time safe.**
    #[inline]
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset all three bands.
    #[inline]
    pub fn reset(&mut self) {
        self.low.reset();
        self.mid.reset();
        self.high.reset();
    }
}

/// A live tone stack chosen at build time — the stateful counterpart of
/// [`crate::pedal::ToneConfig`].
///
/// A [`crate::stage::DriveStage`] owns one of these; `Bypass` lets a pedal
/// that models its tone elsewhere skip the filters entirely.
#[derive(Debug, Clone, Copy)]
pub enum ToneStack {
    /// No tone shaping — a straight passthrough.
    Bypass,
    /// 2-band Baxandall shelving stack.
    Baxandall(Baxandall),
    /// 3-band bass/mid/treble stack.
    ThreeBand(ThreeBand),
}

impl ToneStack {
    /// Filter one sample. **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        match self {
            ToneStack::Bypass => input,
            ToneStack::Baxandall(stack) => stack.process_sample(input),
            ToneStack::ThreeBand(stack) => stack.process_sample(input),
        }
    }

    /// Filter a block in place. **Real-time safe.**
    #[inline]
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        match self {
            ToneStack::Bypass => {}
            ToneStack::Baxandall(stack) => stack.process_block(buffer),
            ToneStack::ThreeBand(stack) => stack.process_block(buffer),
        }
    }

    /// Reset any filter state.
    #[inline]
    pub fn reset(&mut self) {
        match self {
            ToneStack::Bypass => {}
            ToneStack::Baxandall(stack) => stack.reset(),
            ToneStack::ThreeBand(stack) => stack.reset(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;

    /// Convert a linear magnitude to dB.
    fn to_db(magnitude: f32) -> f32 {
        20.0 * magnitude.log10()
    }

    /// A flat (0 dB) shelf must be unity gain everywhere.
    #[test]
    fn flat_low_shelf_is_unity() {
        let biquad = Biquad::low_shelf(120.0, SAMPLE_RATE, 0.0);
        for &freq_frac in &[0.0, 0.01, 0.1, 0.25, 0.49] {
            let gain_db = to_db(biquad.magnitude(freq_frac));
            assert!(
                gain_db.abs() < 0.01,
                "flat shelf not unity at {freq_frac}: {gain_db} dB"
            );
        }
    }

    /// A +6 dB low shelf gives +6 dB at DC and ~0 dB at Nyquist.
    #[test]
    fn low_shelf_boosts_dc() {
        let biquad = Biquad::low_shelf(120.0, SAMPLE_RATE, 6.0);
        let dc_db = to_db(biquad.magnitude(0.0));
        let nyquist_db = to_db(biquad.magnitude(0.5));
        assert!((dc_db - 6.0).abs() < 0.2, "DC gain {dc_db} != 6 dB");
        assert!(nyquist_db.abs() < 0.2, "Nyquist gain {nyquist_db} != 0 dB");
    }

    /// A +6 dB high shelf gives +6 dB at Nyquist and ~0 dB at DC.
    #[test]
    fn high_shelf_boosts_nyquist() {
        let biquad = Biquad::high_shelf(3_000.0, SAMPLE_RATE, 6.0);
        let dc_db = to_db(biquad.magnitude(0.0));
        let nyquist_db = to_db(biquad.magnitude(0.5));
        assert!(dc_db.abs() < 0.2, "DC gain {dc_db} != 0 dB");
        assert!(
            (nyquist_db - 6.0).abs() < 0.2,
            "Nyquist gain {nyquist_db} != 6 dB"
        );
    }

    /// A peaking filter boosts at its centre and is ~flat far away.
    #[test]
    fn peaking_boosts_at_centre() {
        let centre = 1_000.0;
        let biquad = Biquad::peaking(centre, SAMPLE_RATE, 9.0, 1.0);
        let centre_db = to_db(biquad.magnitude(centre / SAMPLE_RATE));
        let dc_db = to_db(biquad.magnitude(0.0));
        assert!(
            (centre_db - 9.0).abs() < 0.3,
            "centre gain {centre_db} != 9 dB"
        );
        assert!(dc_db.abs() < 0.5, "DC gain {dc_db} not ~flat");
    }

    /// Identity biquad is a passthrough.
    #[test]
    fn identity_passes_through() {
        let mut biquad = Biquad::identity();
        for &sample in &[0.0f32, 1.0, -0.5, 0.3] {
            assert!((biquad.process_sample(sample) - sample).abs() < 1e-6);
        }
    }

    /// A flat Baxandall stack passes a signal through at unity RMS.
    #[test]
    fn flat_baxandall_is_transparent() {
        let mut stack = Baxandall::new(SAMPLE_RATE, 0.0, 0.0);
        let mut input_energy = 0.0f64;
        let mut output_energy = 0.0f64;
        for n in 512..8_192 {
            let phase = TAU * 1_000.0 * n as f32 / SAMPLE_RATE;
            let sample = phase.sin();
            // Warm up the state before measuring.
            let filtered = stack.process_sample(sample);
            input_energy += (sample * sample) as f64;
            output_energy += (filtered * filtered) as f64;
        }
        let gain = (output_energy / input_energy).sqrt();
        assert!((gain - 1.0).abs() < 0.05, "flat Baxandall gain {gain} != 1");
    }

    /// Boosting bass raises the low-frequency response of the Baxandall stack.
    #[test]
    fn baxandall_bass_boost_raises_low_end() {
        // Compare the settled DC-ish response by feeding a very low tone.
        let mut flat = Baxandall::new(SAMPLE_RATE, 0.0, 0.0);
        let mut boosted = Baxandall::new(SAMPLE_RATE, 9.0, 0.0);
        let freq = 40.0;
        let mut flat_energy = 0.0f64;
        let mut boost_energy = 0.0f64;
        for n in 0..48_000 {
            let phase = TAU * freq * n as f32 / SAMPLE_RATE;
            let sample = phase.sin();
            let flat_out = flat.process_sample(sample);
            let boost_out = boosted.process_sample(sample);
            if n >= 4_096 {
                flat_energy += (flat_out * flat_out) as f64;
                boost_energy += (boost_out * boost_out) as f64;
            }
        }
        assert!(
            boost_energy > flat_energy * 2.0,
            "bass boost did not raise low end: flat={flat_energy}, boosted={boost_energy}"
        );
    }

    /// A flat 3-band stack is transparent, and reset clears its state.
    #[test]
    fn three_band_flat_and_reset() {
        let mut stack = ThreeBand::new(SAMPLE_RATE, 0.0, 0.0, 0.0);
        let mut buffer: Vec<f32> = (0..256).map(|n| (n as f32 * 0.05).sin()).collect();
        let reference = buffer.clone();
        stack.process_block(&mut buffer);
        for (filtered, original) in buffer.iter().zip(reference.iter()) {
            assert!((filtered - original).abs() < 0.05);
        }
        stack.reset();
    }
}
