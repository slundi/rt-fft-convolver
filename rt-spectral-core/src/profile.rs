//! Noise profile: per-bin noise **power** estimate plus the lock-free hand-off
//! that publishes a freshly-learned profile to the audio thread.
//!
//! The profile is *captured off the audio thread* (the `Learn` button in the
//! plugin): [`learn_profile`] windows a stretch of noise-only input, runs the
//! same analysis FFT the engine uses, and takes the **median magnitude per
//! bin** across frames.  The median (rather than the mean) rejects the
//! occasional non-noise transient that slips into the captured region.
//!
//! The result is squared to a power spectrum because the Wiener denoiser works
//! in the power domain (see [`crate::denoise`]).
//!
//! Hot-swap uses a [`triple_buffer`] channel: [`ProfileSender::publish`] runs
//! off-RT (it may drop the previous profile's allocation), while
//! [`ProfileReceiver::current`] on the audio thread is a wait-free pointer swap
//! with no allocation and no lock.

use realfft::num_complex::Complex;
use rt_fft_convolver::FftHandler;
use triple_buffer::TripleBuffer;

use crate::window::{Window, WindowKind};

/// A per-bin noise **power** spectrum, `N/2 + 1` bins for an `N`-point FFT.
#[derive(Clone, Debug)]
pub struct NoiseProfile {
    /// `|noise(k)|²` per bin.  All-zero means "no noise learned" → the
    /// denoiser passes signal through unchanged.
    power: Vec<f32>,
}

impl NoiseProfile {
    /// A silent (all-zero) profile for `num_bins` spectrum bins.  The denoiser
    /// treats this as pass-through.
    pub fn silent(num_bins: usize) -> Self {
        Self {
            power: vec![0.0; num_bins],
        }
    }

    /// Build a profile directly from a per-bin power slice (mainly for tests).
    pub fn from_power(power: Vec<f32>) -> Self {
        Self { power }
    }

    /// Per-bin noise power, length `N/2 + 1`.
    #[inline]
    pub fn power(&self) -> &[f32] {
        &self.power
    }

    /// Number of spectrum bins (`N/2 + 1`).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.power.len()
    }
}

/// Learn a [`NoiseProfile`] from a stretch of noise-only samples.
///
/// Frames of `fft_size` samples are taken every `hop` samples, Hann-windowed
/// (matching the engine's analysis window), transformed, and the **median**
/// magnitude per bin across all frames is squared into a power spectrum.
///
/// **Allocates** — off-RT only.  Returns a silent profile if `noise` is
/// shorter than one frame.
///
/// # Panics
/// Panics unless `0 < hop <= fft_size` and `fft_size` is even.
pub fn learn_profile(noise: &[f32], fft_size: usize, hop: usize) -> NoiseProfile {
    assert!(hop > 0 && hop <= fft_size, "require 0 < hop <= fft_size");
    let num_bins = fft_size / 2 + 1;
    if noise.len() < fft_size {
        return NoiseProfile::silent(num_bins);
    }

    let window = Window::new(WindowKind::Hann, fft_size);
    let win = window.as_slice();
    let mut fft = FftHandler::new(fft_size);

    let mut frame = vec![0.0f32; fft_size];
    let mut spectrum = vec![Complex::default(); num_bins];

    // Per-bin magnitude samples across frames, transposed on the fly into a
    // bin-major layout so each bin's median is a contiguous sort.
    let num_frames = (noise.len() - fft_size) / hop + 1;
    let mut mags: Vec<Vec<f32>> = vec![Vec::with_capacity(num_frames); num_bins];

    let mut start = 0;
    while start + fft_size <= noise.len() {
        for (dst, (&src, &w)) in frame.iter_mut().zip(noise[start..].iter().zip(win)) {
            *dst = src * w;
        }
        fft.forward(&frame, &mut spectrum);
        for (bin, mag_col) in spectrum.iter().zip(mags.iter_mut()) {
            mag_col.push(bin.norm());
        }
        start += hop;
    }

    let power = mags
        .iter_mut()
        .map(|col| {
            let median = median_in_place(col);
            median * median // magnitude → power
        })
        .collect();

    NoiseProfile { power }
}

/// Median of `values`, computed by sorting in place (off-RT, `values` is a
/// throwaway scratch column).  Returns `0.0` for an empty slice.
fn median_in_place(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// Off-RT writer half of the profile hot-swap channel.
pub struct ProfileSender {
    input: triple_buffer::Input<NoiseProfile>,
}

impl ProfileSender {
    /// Publish a new profile to the audio thread.  **Off-RT**: dropping the
    /// previously-buffered profile may deallocate on this thread.
    #[inline]
    pub fn publish(&mut self, profile: NoiseProfile) {
        self.input.write(profile);
    }
}

/// Audio-thread reader half of the profile hot-swap channel.
pub struct ProfileReceiver {
    output: triple_buffer::Output<NoiseProfile>,
}

impl ProfileReceiver {
    /// The latest published profile.  **Real-time safe**: a wait-free index
    /// swap, no allocation, no lock.
    #[inline]
    pub fn current(&mut self) -> &NoiseProfile {
        self.output.read()
    }
}

/// Create a single-producer / single-consumer profile channel seeded with
/// `initial`.  The `Sender` lives on the capture thread, the `Receiver` on the
/// audio thread.
///
/// **Allocates** — off-RT only (pre-sizes all three internal buffers to the
/// initial profile so no later publish reallocates on the reader).
pub fn profile_channel(initial: NoiseProfile) -> (ProfileSender, ProfileReceiver) {
    let (input, output) = TripleBuffer::new(&initial).split();
    (ProfileSender { input }, ProfileReceiver { output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_profile_is_zero_power() {
        let p = NoiseProfile::silent(1025);
        assert_eq!(p.num_bins(), 1025);
        assert!(p.power().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median_in_place(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_in_place(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_in_place(&mut []), 0.0);
    }

    #[test]
    fn short_input_yields_silent_profile() {
        let p = learn_profile(&[0.1; 100], 256, 64);
        assert_eq!(p.num_bins(), 129);
        assert!(p.power().iter().all(|&x| x == 0.0));
    }

    /// A pure DC noise floor concentrates power in the DC bin; higher bins
    /// stay near zero.
    #[test]
    fn learn_dc_puts_power_in_dc_bin() {
        let fft = 256;
        let noise = vec![0.5f32; fft * 8];
        let p = learn_profile(&noise, fft, fft / 4);
        assert!(p.power()[0] > 0.0, "DC bin should hold power");
        // High bins carry essentially no energy for a constant input.
        assert!(p.power()[fft / 2] < 1e-3);
    }

    /// Publishing a profile makes it visible to the receiver's next read.
    #[test]
    fn hot_swap_delivers_latest_profile() {
        let (mut tx, mut rx) = profile_channel(NoiseProfile::silent(5));
        assert!(rx.current().power().iter().all(|&x| x == 0.0));

        tx.publish(NoiseProfile::from_power(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        assert_eq!(rx.current().power(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
