//! Multi-voice harmoniser: one input note → a chord of pitch-shifted voices.
//!
//! The harmoniser owns up to [`MAX_VOICES`] [`Voice`]s and a
//! [`VoicePitchTracker`] that follows the incoming (monophonic) pitch.  Each
//! voice's interval is sourced according to the active [`SnapMode`]:
//!
//! - [`SnapMode::Free`]  — the voice's configured fixed offset.
//! - [`SnapMode::Scale`] — that offset applied to the tracked note and snapped
//!   to the nearest degree of the configured [`Scale`].
//! - [`SnapMode::Midi`]  — the interval between the tracked note and a held MIDI
//!   note assigned to the voice.
//!
//! Note-on / note-off only mutate fixed-size arrays, so MIDI can drive the
//! voices sample-accurately from inside `process` without allocating (Roadmap-08
//! RT-safety requirement).  Output is **wet only** — the dry/wet `Mix` control
//! lives in the Phase-2 plugin.

use rt_pitch_core::{PhaseVocoder, window::WindowKind};

use crate::pitch::VoicePitchTracker;
use crate::voice::Voice;
use crate::voicing::{Scale, SnapMode, hz_to_midi, midi_to_semitone_offset};

/// Maximum number of simultaneous harmony voices (Roadmap-08 Phase 2: 4).
pub const MAX_VOICES: usize = 4;

/// Voice manager that renders a chord of pitch-shifted voices from one input.
pub struct Harmoniser {
    voices: Vec<Voice>,
    /// Fixed per-voice semitone offset (used by `Free` and `Scale`).
    free_intervals: [f32; MAX_VOICES],
    /// Whether a voice participates in `Free` / `Scale` modes.
    enabled: [bool; MAX_VOICES],
    /// MIDI note assigned to each voice slot (used by `Midi` mode).
    held_notes: [Option<u8>; MAX_VOICES],
    snap_mode: SnapMode,
    scale: Scale,
    tracker: VoicePitchTracker,
    /// Last tracked fundamental in Hz, cached for UI / interval maths.
    detected_hz: Option<f32>,
}

impl Harmoniser {
    /// Build a harmoniser at `sample_rate` rendering host blocks up to
    /// `max_block`.  Every voice defaults to a formant-preserving phase vocoder.
    ///
    /// **Allocates** — construct off the audio thread.
    pub fn new(sample_rate: f32, max_block: usize) -> Self {
        // 1024-point phase vocoders: chord-capable and formant-aware, the
        // Roadmap-08 default back-end for harmony voices.
        let voices = (0..MAX_VOICES)
            .map(|_| {
                let shifter = Box::new(PhaseVocoder::new(1024, max_block, WindowKind::Hann));
                Voice::new(shifter, max_block)
            })
            .collect();

        Self {
            voices,
            free_intervals: [0.0; MAX_VOICES],
            enabled: [false; MAX_VOICES],
            held_notes: [None; MAX_VOICES],
            snap_mode: SnapMode::default(),
            scale: Scale::default(),
            // Track voice fundamentals from ~65 Hz (C2) to 1 kHz, updating
            // every 512 samples.
            tracker: VoicePitchTracker::new(sample_rate, 65.0, 1000.0, 512, max_block),
            detected_hz: None,
        }
    }

    /// Select where voice intervals come from.
    #[inline]
    pub fn set_snap_mode(&mut self, mode: SnapMode) {
        self.snap_mode = mode;
    }

    /// Set the scale used by [`SnapMode::Scale`].
    #[inline]
    pub fn set_scale(&mut self, scale: Scale) {
        self.scale = scale;
    }

    /// Configure a voice's fixed offset (semitones) and enable it for the
    /// `Free` / `Scale` modes.
    ///
    /// # Panics
    /// Panics if `index >= MAX_VOICES`.
    pub fn set_voice_interval(&mut self, index: usize, semitones: f32) {
        self.free_intervals[index] = semitones;
        self.enabled[index] = true;
    }

    /// Enable or disable a voice for the `Free` / `Scale` modes.
    ///
    /// # Panics
    /// Panics if `index >= MAX_VOICES`.
    pub fn set_voice_enabled(&mut self, index: usize, enabled: bool) {
        self.enabled[index] = enabled;
    }

    /// Mutable access to a voice's mix parameters (level / pan / detune).
    ///
    /// # Panics
    /// Panics if `index >= MAX_VOICES`.
    #[inline]
    pub fn voice_mut(&mut self, index: usize) -> &mut Voice {
        &mut self.voices[index]
    }

    /// Register a held MIDI note (assigns it to the first free voice slot).
    /// Ignored if all slots are occupied.  RT-safe.
    pub fn note_on(&mut self, note: u8) {
        if self.held_notes.contains(&Some(note)) {
            return; // already held
        }
        if let Some(slot) = self.held_notes.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(note);
        }
    }

    /// Release a held MIDI note.  RT-safe.
    pub fn note_off(&mut self, note: u8) {
        for slot in &mut self.held_notes {
            if *slot == Some(note) {
                *slot = None;
            }
        }
    }

    /// Release every held MIDI note.
    pub fn all_notes_off(&mut self) {
        self.held_notes = [None; MAX_VOICES];
    }

    /// The most recently tracked input fundamental in Hz (for UI / debugging).
    #[inline]
    pub fn detected_hz(&self) -> Option<f32> {
        self.detected_hz
    }

    /// Algorithmic latency in samples (the per-voice shifter latency).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.voices.first().map_or(0, Voice::latency_samples)
    }

    /// Render one block: `input` (dry voice) → `left` / `right` wet harmony.
    ///
    /// The output buffers are overwritten (not accumulated into).  All three
    /// slices share the input length (`<= max_block`).  Allocation-free.
    pub fn process_block(&mut self, input: &[f32], left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(input.len(), left.len());
        debug_assert_eq!(input.len(), right.len());
        left.fill(0.0);
        right.fill(0.0);

        // Update the pitch estimate for this block once.
        if let Some(estimate) = self.tracker.process_block(input) {
            self.detected_hz = Some(estimate.frequency);
        }

        for index in 0..self.voices.len() {
            if let Some(semitones) = self.interval_for(index) {
                self.voices[index].set_interval_semitones(semitones);
                self.voices[index].process_add(input, left, right);
            }
        }
    }

    /// Resolve the interval (semitones) for voice `index` under the active
    /// [`SnapMode`], or `None` if the voice is inactive this block.
    fn interval_for(&self, index: usize) -> Option<f32> {
        match self.snap_mode {
            SnapMode::Free => self.enabled[index].then(|| self.free_intervals[index]),
            SnapMode::Scale => {
                if !self.enabled[index] {
                    return None;
                }
                let input_midi = hz_to_midi(self.detected_hz?);
                let target = self
                    .scale
                    .snap_midi(input_midi + self.free_intervals[index]);
                Some(target - input_midi)
            }
            SnapMode::Midi => {
                let note = self.held_notes[index]?;
                Some(midi_to_semitone_offset(note, self.detected_hz?))
            }
        }
    }

    /// Flush every voice and the pitch tracker.
    pub fn reset(&mut self) {
        for voice in &mut self.voices {
            voice.reset();
        }
        self.tracker.reset();
        self.detected_hz = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dominant_frequency_hz, make_sawtooth, rms};
    use crate::voicing::ScaleKind;

    const SAMPLE_RATE: f32 = 48_000.0;
    const MAX_BLOCK: usize = 512;

    fn run(harm: &mut Harmoniser, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut left = Vec::with_capacity(input.len());
        let mut right = Vec::with_capacity(input.len());
        let mut lbuf = vec![0.0f32; MAX_BLOCK];
        let mut rbuf = vec![0.0f32; MAX_BLOCK];
        for chunk in input.chunks(MAX_BLOCK) {
            let len = chunk.len();
            harm.process_block(chunk, &mut lbuf[..len], &mut rbuf[..len]);
            left.extend_from_slice(&lbuf[..len]);
            right.extend_from_slice(&rbuf[..len]);
        }
        (left, right)
    }

    #[test]
    fn no_active_voices_is_silent() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Midi); // no notes held
        let input = make_sawtooth(220.0, SAMPLE_RATE, 16_000);
        let (left, right) = run(&mut harm, &input);
        assert!(rms(&left) < 1e-6 && rms(&right) < 1e-6);
    }

    #[test]
    fn free_unison_preserves_pitch() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Free);
        harm.set_voice_interval(0, 0.0); // unison
        let input = make_sawtooth(220.0, SAMPLE_RATE, 20_000);
        let (left, right) = run(&mut harm, &input);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(l, r)| l + r).collect();
        let freq = dominant_frequency_hz(&mono, SAMPLE_RATE);
        assert!((freq - 220.0).abs() < 5.0, "unison freq = {freq}");
    }

    #[test]
    fn free_octave_voice_shifts_up() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Free);
        harm.set_voice_interval(0, 12.0); // one octave up
        let input = make_sawtooth(220.0, SAMPLE_RATE, 20_000);
        let (left, right) = run(&mut harm, &input);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(l, r)| l + r).collect();
        let freq = dominant_frequency_hz(&mono, SAMPLE_RATE);
        assert!((freq - 440.0).abs() < 10.0, "octave-up freq = {freq}");
    }

    #[test]
    fn midi_note_drives_a_voice() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Midi);
        // Input A3 (220 Hz ≈ MIDI 57); hold A4 (MIDI 69) → +12 → 440 Hz.
        harm.note_on(69);
        let input = make_sawtooth(220.0, SAMPLE_RATE, 20_000);
        let (left, right) = run(&mut harm, &input);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(l, r)| l + r).collect();
        let freq = dominant_frequency_hz(&mono, SAMPLE_RATE);
        assert!((freq - 440.0).abs() < 12.0, "midi voice freq = {freq}");
    }

    #[test]
    fn note_off_silences_the_voice() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Midi);
        harm.note_on(69);
        harm.note_off(69);
        let input = make_sawtooth(220.0, SAMPLE_RATE, 8_000);
        let (left, right) = run(&mut harm, &input);
        assert!(rms(&left) < 1e-6 && rms(&right) < 1e-6);
    }

    #[test]
    fn scale_mode_snaps_into_key() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Scale);
        harm.set_scale(Scale::new(0, ScaleKind::Major)); // C major
        // A "third above" request (+4) on various inputs should stay diatonic;
        // here just verify the engine produces audible, finite output.
        harm.set_voice_interval(0, 4.0);
        let input = make_sawtooth(261.63, SAMPLE_RATE, 20_000); // C4
        let (left, _right) = run(&mut harm, &input);
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(rms(&left[4096..]) > 1e-5, "scale voice should be audible");
    }

    #[test]
    fn note_on_is_idempotent_and_bounded() {
        let mut harm = Harmoniser::new(SAMPLE_RATE, MAX_BLOCK);
        harm.set_snap_mode(SnapMode::Midi);
        // Hold the same note twice, then fill every slot and one extra.
        harm.note_on(60);
        harm.note_on(60);
        harm.note_on(62);
        harm.note_on(64);
        harm.note_on(65);
        harm.note_on(67); // one past MAX_VOICES → ignored, no panic
        // Only MAX_VOICES notes can be held.
        let held = (0..128u8).filter(|&n| {
            // Re-registering a held note is a no-op, so count distinct slots.
            harm.held_notes.contains(&Some(n))
        });
        assert_eq!(held.count(), MAX_VOICES);
    }
}
