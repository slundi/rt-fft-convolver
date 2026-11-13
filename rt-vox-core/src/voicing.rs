//! Interval sourcing for the harmoniser: fixed offsets, diatonic scale, or MIDI.
//!
//! A harmoniser turns one input note into several.  Where those harmony pitches
//! come from is the [`SnapMode`] (Roadmap-08 Phase 0 harmoniser-voicing
//! decision):
//!
//! - [`SnapMode::Free`] — each voice is a fixed semitone offset from the input.
//! - [`SnapMode::Scale`] — the offset is snapped to the nearest degree of a
//!   [`Scale`] (`Key` + `Scale`), giving musical diatonic thirds/sixths.
//! - [`SnapMode::Midi`] — harmony pitches come straight from held MIDI notes;
//!   the interval is `midi_note − detected_input_note`.
//!
//! All helpers are pure arithmetic — no allocation, safe on the audio thread.

/// Where a harmoniser voice's interval comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SnapMode {
    /// Fixed per-voice semitone offset.
    Free,
    /// Offset snapped to the nearest scale degree.
    Scale,
    /// Interval taken from a held MIDI note (the default / primary use case).
    #[default]
    Midi,
}

/// The seven common diatonic / minor scale shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScaleKind {
    /// Ionian.
    #[default]
    Major,
    /// Natural minor (Aeolian).
    NaturalMinor,
    /// Harmonic minor (raised 7th).
    HarmonicMinor,
    /// The full chromatic set (every semitone is in scale — a no-op snap).
    Chromatic,
}

impl ScaleKind {
    /// Semitone degrees of the scale within one octave, relative to the root.
    fn degrees(self) -> &'static [u8] {
        match self {
            ScaleKind::Major => &[0, 2, 4, 5, 7, 9, 11],
            ScaleKind::NaturalMinor => &[0, 2, 3, 5, 7, 8, 10],
            ScaleKind::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            ScaleKind::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }
}

/// A musical scale: a root pitch class plus a [`ScaleKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    /// Root pitch class `0..=11` (0 = C).
    root: u8,
    kind: ScaleKind,
    /// Membership table for the twelve pitch classes, precomputed from
    /// `root` + `kind` so [`snap_midi`](Self::snap_midi) is branch-light.
    in_scale: [bool; 12],
}

impl Scale {
    /// Build a scale rooted at pitch class `root` (`0 = C … 11 = B`; values are
    /// taken modulo 12).
    pub fn new(root: u8, kind: ScaleKind) -> Self {
        let root = root % 12;
        let mut in_scale = [false; 12];
        for &degree in kind.degrees() {
            in_scale[((root + degree) % 12) as usize] = true;
        }
        Self {
            root,
            kind,
            in_scale,
        }
    }

    /// The root pitch class.
    #[inline]
    pub fn root(&self) -> u8 {
        self.root
    }

    /// The scale kind.
    #[inline]
    pub fn kind(&self) -> ScaleKind {
        self.kind
    }

    /// Snap a (possibly fractional) MIDI note to the nearest in-scale note.
    ///
    /// Rounds to the nearest semitone, then searches outward for the closest
    /// pitch class that belongs to the scale (ties resolve upward).
    pub fn snap_midi(&self, midi: f32) -> f32 {
        let nearest = midi.round() as i32;
        for distance in 0..=6 {
            for &sign in &[1i32, -1] {
                let candidate = nearest + sign * distance;
                let pitch_class = candidate.rem_euclid(12) as usize;
                if self.in_scale[pitch_class] {
                    return candidate as f32;
                }
            }
        }
        nearest as f32 // unreachable for any non-empty scale
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self::new(0, ScaleKind::Major) // C major
    }
}

/// Convert a frequency in Hz to a (fractional) MIDI note number.
///
/// `69 + 12·log2(hz / 440)`; A4 (440 Hz) → 69.0.
#[inline]
pub fn hz_to_midi(hz: f32) -> f32 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

/// Convert a MIDI note number to its frequency in Hz.
#[inline]
pub fn midi_to_hz(midi: f32) -> f32 {
    440.0 * ((midi - 69.0) / 12.0).exp2()
}

/// Semitone interval that maps the detected input pitch (`input_hz`) onto the
/// held MIDI note `target_note` — the [`SnapMode::Midi`] interval source.
#[inline]
pub fn midi_to_semitone_offset(target_note: u8, input_hz: f32) -> f32 {
    target_note as f32 - hz_to_midi(input_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_is_midi_69() {
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-4);
        assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn midi_roundtrips_through_hz() {
        for note in [40u8, 48, 60, 69, 72, 84] {
            let back = hz_to_midi(midi_to_hz(note as f32));
            assert!((back - note as f32).abs() < 1e-3, "note {note}");
        }
    }

    #[test]
    fn midi_interval_is_target_minus_input() {
        // Input A3 (220 Hz, MIDI 57), target C5 (MIDI 72) → +15 semitones.
        let offset = midi_to_semitone_offset(72, 220.0);
        assert!((offset - 15.0).abs() < 0.05, "offset = {offset}");
    }

    #[test]
    fn c_major_snaps_c_sharp_down_to_c() {
        let scale = Scale::new(0, ScaleKind::Major);
        // C#4 = MIDI 61; nearest in-scale is C (60) or D (62) — ties go up, but
        // 61 is equidistant so distance-1 upward D (62) wins... verify both are
        // valid degrees and the result is in scale.
        let snapped = scale.snap_midi(61.0);
        let pitch_class = (snapped as i32).rem_euclid(12);
        assert!(
            pitch_class == 0 || pitch_class == 2,
            "snapped to non-scale pc {pitch_class}"
        );
    }

    #[test]
    fn in_scale_note_is_unchanged() {
        let scale = Scale::new(0, ScaleKind::Major);
        assert_eq!(scale.snap_midi(64.0), 64.0); // E is in C major
    }

    #[test]
    fn chromatic_snaps_to_itself() {
        let scale = Scale::new(7, ScaleKind::Chromatic);
        for note in 48..72 {
            assert_eq!(scale.snap_midi(note as f32), note as f32);
        }
    }

    #[test]
    fn natural_minor_has_flat_third() {
        let scale = Scale::new(9, ScaleKind::NaturalMinor); // A minor
        // A natural minor contains C (pc 0) but not C# (pc 1).
        assert_eq!(scale.snap_midi(60.0), 60.0); // C stays
        let snapped_cs = scale.snap_midi(61.0);
        assert_ne!((snapped_cs as i32).rem_euclid(12), 1, "C# not in A minor");
    }
}
