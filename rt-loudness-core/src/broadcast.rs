//! Lock-free meter-value broadcast (audio thread → UI / logger).
//!
//! The audio thread produces new readings every block; the editor and the
//! off-RT CSV/JSON logger consume the *latest* set whenever they wake.  A
//! channel or mutex would either block the audio thread or queue stale values,
//! so instead each scalar is published through an `AtomicU32` holding the
//! `f32` bit pattern.  Writers `store` and readers `load` with `Relaxed`
//! ordering: the reader may occasionally see one value from block *n* next to
//! another from block *n+1* (a "torn" set), which is harmless for a meter
//! readout that refreshes tens of times a second.
//!
//! # Real-time safety
//!
//! [`MeterBroadcast::store`] is seven relaxed atomic writes — no allocation,
//! no lock, wait-free.

use std::sync::atomic::{AtomicU32, Ordering};

/// One coherent set of meter readings.
///
/// All loudness fields are LUFS, `LRA` is LU, peaks are dB.  A field is
/// `f32::NEG_INFINITY` when it has no valid value yet (silence / not enough
/// data).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeterValues {
    /// Momentary loudness, 400 ms (LUFS).
    pub momentary_lufs: f32,
    /// Short-term loudness, 3 s (LUFS).
    pub short_term_lufs: f32,
    /// Gated integrated loudness, session (LUFS).
    pub integrated_lufs: f32,
    /// Loudness range (LU).
    pub lra_lu: f32,
    /// True (inter-sample) peak (dBTP).
    pub true_peak_dbtp: f32,
    /// Sample peak (dBFS).
    pub sample_peak_dbfs: f32,
    /// Unweighted RMS over the momentary window (dB).
    pub rms_db: f32,
}

impl MeterValues {
    /// The "nothing measured yet" reading: every value at silence.
    pub const SILENCE: Self = Self {
        momentary_lufs: f32::NEG_INFINITY,
        short_term_lufs: f32::NEG_INFINITY,
        integrated_lufs: f32::NEG_INFINITY,
        lra_lu: 0.0,
        true_peak_dbtp: f32::NEG_INFINITY,
        sample_peak_dbfs: f32::NEG_INFINITY,
        rms_db: f32::NEG_INFINITY,
    };
}

impl Default for MeterValues {
    fn default() -> Self {
        Self::SILENCE
    }
}

/// Lock-free publisher of the latest [`MeterValues`].
///
/// Wrap in an `Arc` and share the same instance between the audio-thread
/// producer and the UI/logger consumers.
#[derive(Debug)]
pub struct MeterBroadcast {
    momentary: AtomicU32,
    short_term: AtomicU32,
    integrated: AtomicU32,
    lra: AtomicU32,
    true_peak: AtomicU32,
    sample_peak: AtomicU32,
    rms: AtomicU32,
}

impl Default for MeterBroadcast {
    fn default() -> Self {
        Self::new()
    }
}

impl MeterBroadcast {
    /// Create a broadcast initialised to silence.
    pub fn new() -> Self {
        let values = MeterValues::SILENCE;
        Self {
            momentary: AtomicU32::new(values.momentary_lufs.to_bits()),
            short_term: AtomicU32::new(values.short_term_lufs.to_bits()),
            integrated: AtomicU32::new(values.integrated_lufs.to_bits()),
            lra: AtomicU32::new(values.lra_lu.to_bits()),
            true_peak: AtomicU32::new(values.true_peak_dbtp.to_bits()),
            sample_peak: AtomicU32::new(values.sample_peak_dbfs.to_bits()),
            rms: AtomicU32::new(values.rms_db.to_bits()),
        }
    }

    /// Publish a new set of readings.  **Real-time safe** (wait-free).
    #[inline]
    pub fn store(&self, values: &MeterValues) {
        self.momentary
            .store(values.momentary_lufs.to_bits(), Ordering::Relaxed);
        self.short_term
            .store(values.short_term_lufs.to_bits(), Ordering::Relaxed);
        self.integrated
            .store(values.integrated_lufs.to_bits(), Ordering::Relaxed);
        self.lra.store(values.lra_lu.to_bits(), Ordering::Relaxed);
        self.true_peak
            .store(values.true_peak_dbtp.to_bits(), Ordering::Relaxed);
        self.sample_peak
            .store(values.sample_peak_dbfs.to_bits(), Ordering::Relaxed);
        self.rms.store(values.rms_db.to_bits(), Ordering::Relaxed);
    }

    /// Read the latest published readings.
    #[inline]
    pub fn load(&self) -> MeterValues {
        MeterValues {
            momentary_lufs: f32::from_bits(self.momentary.load(Ordering::Relaxed)),
            short_term_lufs: f32::from_bits(self.short_term.load(Ordering::Relaxed)),
            integrated_lufs: f32::from_bits(self.integrated.load(Ordering::Relaxed)),
            lra_lu: f32::from_bits(self.lra.load(Ordering::Relaxed)),
            true_peak_dbtp: f32::from_bits(self.true_peak.load(Ordering::Relaxed)),
            sample_peak_dbfs: f32::from_bits(self.sample_peak.load(Ordering::Relaxed)),
            rms_db: f32::from_bits(self.rms.load(Ordering::Relaxed)),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_silence() {
        let broadcast = MeterBroadcast::new();
        assert_eq!(broadcast.load(), MeterValues::SILENCE);
    }

    #[test]
    fn round_trips_values() {
        let broadcast = MeterBroadcast::new();
        let values = MeterValues {
            momentary_lufs: -14.2,
            short_term_lufs: -15.0,
            integrated_lufs: -23.0,
            lra_lu: 7.5,
            true_peak_dbtp: -0.3,
            sample_peak_dbfs: -1.1,
            rms_db: -18.6,
        };
        broadcast.store(&values);
        assert_eq!(broadcast.load(), values);
    }

    #[test]
    fn preserves_negative_infinity() {
        let broadcast = MeterBroadcast::new();
        let mut values = MeterValues::SILENCE;
        values.lra_lu = 3.0;
        broadcast.store(&values);
        let read = broadcast.load();
        assert_eq!(read.momentary_lufs, f32::NEG_INFINITY);
        assert_eq!(read.lra_lu, 3.0);
    }
}
