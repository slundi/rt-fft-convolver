//! The polyphonic voice manager — a fixed-capacity pool of [`Voice`]s with
//! allocation-free note-on / note-off / voice-stealing (Phase 1 "Voice manager"
//! and the Phase 0 polyphony decision in
//! `docs/Roadmap-06-physical-modeling.md`).
//!
//! # Fixed capacity
//!
//! The pool is a `[Voice; N]` allocated once at construction (each voice's DSP
//! buffers with it).  Nothing is allocated afterwards, so `note_on` is safe on
//! the render thread even though it may retune a string.  Phase 0 fixed `N` at
//! 16 ([`crate::MAX_VOICES`]); the `Polyphony` control caps how many of the 16
//! may sound via [`set_polyphony`](VoiceManager::set_polyphony) rather than
//! resizing the pool.
//!
//! # Voice stealing: quietest, then oldest
//!
//! On a note-on the manager scores every eligible voice and reuses the lowest
//! score:
//!
//! ```text
//! score = level  + (held ? STEAL_PENALTY : 0)  + serial · SERIAL_WEIGHT
//! ```
//!
//! - A free, silent voice scores ≈ 0 and is taken first.
//! - Among held voices, the **quietest** (lowest `level`) wins — least audible
//!   to cut — with the tiny `serial` term breaking ties toward the **oldest**.
//!
//! This is the Phase 0 rule in one pass, no sorting, no allocation.
//!
//! # Real-time safety
//!
//! [`process`](VoiceManager::process), [`note_on`](VoiceManager::note_on) and
//! [`note_off`](VoiceManager::note_off) allocate nothing, lock nothing and do
//! not `panic` in release builds.  Change the model family by constructing a
//! new manager off the render thread.

use crate::pitch::max_loop_len;
use crate::voice::{PhysicalParams, Voice};

/// Level below which an unheld voice counts as silent / reusable.
const SILENCE: f32 = 1.0e-4;

/// Added to a held voice's score so held notes are stolen only after every free
/// or released voice has been used.
const STEAL_PENALTY: f32 = 1.0e6;

/// Weight of the age term — small enough not to disturb the level ordering, big
/// enough to break ties toward the oldest voice.
const SERIAL_WEIGHT: f32 = 1.0e-9;

/// A fixed pool of `N` voices with RT-safe allocation and stealing.
#[derive(Debug, Clone)]
pub struct VoiceManager<const N: usize> {
    voices: [Voice; N],
    params: PhysicalParams,
    sample_rate: f32,
    /// Monotonic counter stamped onto each triggered voice (age ordering).
    serial: u64,
    /// How many of the `N` voices may currently sound (`Polyphony` control).
    active_limit: usize,
}

impl<const N: usize> VoiceManager<N> {
    /// Build a pool of `N` voices for `params.model` at `sample_rate`.
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(sample_rate: f32, params: PhysicalParams) -> Self {
        let max_len = max_loop_len(sample_rate);
        let model = params.model;
        let voices = core::array::from_fn(|_| Voice::new(sample_rate, model, max_len));
        Self {
            voices,
            params,
            sample_rate,
            serial: 0,
            active_limit: N,
        }
    }

    /// Total voice capacity (`N`).
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Update the per-note parameters applied to *future* triggers.  The
    /// `model` field is ignored here — switching family requires a new manager
    /// (the kernels are fixed at construction).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_params(&mut self, params: PhysicalParams) {
        let model = self.params.model; // preserve the constructed family
        self.params = params;
        self.params.model = model;
    }

    /// Cap how many voices may sound (the `Polyphony` control), clamped to
    /// `[1, N]`.  Already-ringing voices above the new cap keep decaying.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_polyphony(&mut self, limit: usize) {
        self.active_limit = limit.clamp(1, N);
    }

    /// The current polyphony cap.
    #[inline]
    pub fn polyphony(&self) -> usize {
        self.active_limit
    }

    /// Number of voices currently producing audible sound (for a CPU / activity
    /// meter).
    #[inline]
    pub fn active_count(&self) -> usize {
        self.voices
            .iter()
            .filter(|voice| voice.level() >= SILENCE)
            .count()
    }

    /// Start `note` at `velocity ∈ [0, 1]`, allocating or stealing a voice.
    ///
    /// **Real-time safe.**
    pub fn note_on(&mut self, note: u8, velocity: f32) {
        let limit = self.active_limit.clamp(1, N);

        // Pick the lowest-scoring eligible voice (immutable pass, then mutate).
        let mut best_index = 0;
        let mut best_score = f32::INFINITY;
        for (index, voice) in self.voices.iter().take(limit).enumerate() {
            let held = if voice.is_held() { STEAL_PENALTY } else { 0.0 };
            let score = voice.level() + held + voice.serial() as f32 * SERIAL_WEIGHT;
            if score < best_score {
                best_score = score;
                best_index = index;
            }
        }

        self.serial += 1;
        let serial = self.serial;
        let params = self.params;
        self.voices[best_index].trigger(note, velocity, serial, &params, self.sample_rate);
    }

    /// Release every held voice matching `note` (a note can be doubled across
    /// voices if retriggered before release).
    ///
    /// **Real-time safe.**
    pub fn note_off(&mut self, note: u8) {
        for voice in &mut self.voices {
            if voice.is_held() && voice.note() == note {
                voice.release();
            }
        }
    }

    /// Silence and free every voice.
    pub fn reset(&mut self) {
        for voice in &mut self.voices {
            voice.reset();
        }
        self.serial = 0;
    }

    /// Sum one sample from every voice.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self) -> f32 {
        let mut sum = 0.0;
        for voice in &mut self.voices {
            sum += voice.process();
        }
        sum
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::{Model, PhysicalParams};

    const SR: f32 = 48_000.0;

    fn plucked_manager() -> VoiceManager<16> {
        VoiceManager::new(
            SR,
            PhysicalParams {
                model: Model::Plucked,
                ..PhysicalParams::default()
            },
        )
    }

    #[test]
    fn note_on_produces_sound() {
        let mut manager = plucked_manager();
        manager.note_on(69, 1.0);
        let mut peak = 0.0f32;
        for _ in 0..4_000 {
            peak = peak.max(manager.process().abs());
        }
        assert!(peak > 1e-3, "note-on made no sound");
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn capacity_and_polyphony_clamp() {
        let mut manager = plucked_manager();
        assert_eq!(manager.capacity(), 16);
        manager.set_polyphony(100);
        assert_eq!(manager.polyphony(), 16);
        manager.set_polyphony(0);
        assert_eq!(manager.polyphony(), 1);
    }

    #[test]
    fn distinct_notes_use_distinct_voices() {
        let mut manager = plucked_manager();
        for note in 60..68 {
            manager.note_on(note, 1.0);
            manager.process();
        }
        // Eight fresh notes → eight voices with energy (well under the 16 cap).
        for _ in 0..1_000 {
            manager.process();
        }
        assert_eq!(manager.active_count(), 8);
    }

    #[test]
    fn stealing_never_exceeds_capacity() {
        let mut manager = plucked_manager();
        // Fire far more notes than voices, all held.
        for note in 40..90 {
            manager.note_on(note, 1.0);
            manager.process();
        }
        for _ in 0..1_000 {
            let sample = manager.process();
            assert!(sample.is_finite(), "steal produced non-finite output");
        }
        assert!(manager.active_count() <= manager.capacity());
    }

    #[test]
    fn quietest_voice_is_stolen_first() {
        // Two-voice pool: one loud (fresh), one quiet (well-decayed).
        let mut manager: VoiceManager<2> = VoiceManager::new(
            SR,
            PhysicalParams {
                model: Model::Plucked,
                t60: 0.3, // fast enough that voice 0 goes quiet within a second
                ..PhysicalParams::default()
            },
        );
        manager.note_on(80, 1.0); // voice 0
        for _ in 0..SR as usize {
            manager.process(); // ~1 s: let voice 0 decay to near silence
        }
        manager.note_on(50, 1.0); // voice 1, fresh & loud
        // Run long enough for the low note's delay line to fill and ring loudly.
        for _ in 0..1_500 {
            manager.process();
        }
        let quiet_level = manager.voices[0].level();
        let loud_level = manager.voices[1].level();
        assert!(
            quiet_level < loud_level,
            "test setup: voice 0 should be quieter"
        );

        // A third note must steal the quiet voice (0), not the loud one (1).
        manager.note_on(60, 1.0);
        assert_eq!(
            manager.voices[0].note(),
            60,
            "did not steal the quietest voice"
        );
        assert_eq!(manager.voices[1].note(), 50, "wrongly stole the loud voice");
    }

    #[test]
    fn oldest_breaks_ties_between_equally_quiet_voices() {
        // Fresh pool: all voices silent and unheld, serial 0 → identical scores
        // except the age term. The first note takes voice 0; with all still
        // "available" the manager should keep filling in order.
        let mut manager: VoiceManager<4> = VoiceManager::new(
            SR,
            PhysicalParams {
                model: Model::Plucked,
                ..PhysicalParams::default()
            },
        );
        manager.note_on(60, 1.0);
        assert_eq!(manager.voices[0].note(), 60);
        manager.note_on(62, 1.0);
        assert_eq!(
            manager.voices[1].note(),
            62,
            "second note should fill next slot"
        );
    }

    #[test]
    fn note_off_releases_matching_voice() {
        let mut manager = plucked_manager();
        manager.note_on(69, 1.0);
        assert!(manager.voices[0].is_held());
        manager.note_off(69);
        assert!(!manager.voices[0].is_held());
    }

    #[test]
    fn reset_silences_all_voices() {
        let mut manager = plucked_manager();
        for note in 60..70 {
            manager.note_on(note, 1.0);
        }
        for _ in 0..500 {
            manager.process();
        }
        manager.reset();
        assert_eq!(manager.active_count(), 0);
        // Output is silent right after reset.
        for _ in 0..64 {
            assert!(manager.process().abs() < 1e-6);
        }
    }
}
