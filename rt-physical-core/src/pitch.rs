//! MIDI ↔ frequency conversion for the voice layer.

/// Lowest fundamental the engine is built to play.  Delay lines and combs are
/// sized from this: `capacity = ceil(fs / MIN_FREQ_HZ)`.  20 Hz sits just below
/// the bottom of hearing and well below a bass guitar's low B (≈ 31 Hz).
pub const MIN_FREQ_HZ: f32 = 20.0;

/// Convert a MIDI note number to its equal-tempered frequency in Hz
/// (`440·2^((note − 69)/12)`, A4 = MIDI 69 = 440 Hz).
#[inline]
pub fn midi_to_freq(note: u8) -> f32 {
    440.0 * ((note as f32 - 69.0) / 12.0).exp2()
}

/// Delay-line / comb capacity (samples) needed to reach [`MIN_FREQ_HZ`] at the
/// given sample rate, with a little headroom.
#[inline]
pub fn max_loop_len(sample_rate: f32) -> usize {
    (sample_rate / MIN_FREQ_HZ).ceil() as usize + 4
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_is_440() {
        assert!((midi_to_freq(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn octaves_double() {
        assert!((midi_to_freq(81) - 880.0).abs() < 1e-3); // A5
        assert!((midi_to_freq(57) - 220.0).abs() < 1e-3); // A3
    }

    #[test]
    fn capacity_covers_the_lowest_note() {
        let sample_rate = 48_000.0;
        let capacity = max_loop_len(sample_rate);
        assert!(capacity as f32 >= sample_rate / MIN_FREQ_HZ);
    }
}
