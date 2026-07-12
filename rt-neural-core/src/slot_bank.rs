//! [`ModelSlotBank`] — a fixed bank of `N` neural-model slots with a
//! deferred activation intent.
//!
//! # Model
//!
//! The bank owns `N` slots (default 8, matching the roadmap and mirroring
//! `rt-ir-plugin-core`'s `IrSlotBank`).  Each slot is either empty or
//! holds a [`PreparedModel`].  Loading and slot swaps are triggered from
//! the control thread; the audio thread only ever needs to know which
//! slot is currently active.
//!
//! Activation is an **intent** — the control thread flags "please switch
//! to slot 3 at the next block boundary" and the audio thread consumes
//! that intent at the top of its next callback.  This keeps every audio
//! block internally consistent and avoids mid-block state changes.
//!
//! ```text
//! control thread:  load(3, model)         request_activation(3)
//!                        │                          │
//!                        ▼                          ▼
//!                  slots[3] = Some(m)         pending = Some(3)
//!                                                    │  (next block)
//! audio thread:                          take_pending() ─┴─► active = 3
//! ```
//!
//! # Empty slots are legal
//!
//! Per the neural-amp roadmap, "empty slots pass audio through dry — never
//! panic".  [`request_activation`](ModelSlotBank::request_activation)
//! therefore accepts any in-range index, loaded or not; the audio thread
//! decides how to render an empty slot (typically dry pass-through).
//! Only an out-of-range index returns [`SlotError`].
//!
//! # No cross-thread sync inside the bank
//!
//! `ModelSlotBank` is a plain state container — thread-safe access is
//! delegated to a handoff primitive owned by the plugin.  Phase 2 will
//! wire a `triple_buffer::Buffer<Option<PreparedModel>>` or an
//! `arc_swap::ArcSwap` around the active slot's model; the bank itself
//! stays synchronous and trivially testable.

use std::array;

use crate::prepared::PreparedModel;

/// Number of independently swappable model slots.  Matches the neural-amp
/// roadmap and the sibling `IrSlotBank` used by the IR plugins.
pub const SLOT_COUNT: usize = 8;

/// Error returned by fallible [`ModelSlotBank`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotError {
    /// The requested slot index was `>= N`.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The bank capacity `N`.
        capacity: usize,
    },
}

impl std::fmt::Display for SlotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotError::IndexOutOfRange { index, capacity } => write!(
                formatter,
                "slot index {index} out of range (capacity {capacity})"
            ),
        }
    }
}

impl std::error::Error for SlotError {}

/// A bank of `N` [`PreparedModel`] slots plus a single-slot "activate on
/// next block" intent.
///
/// `N` defaults to 8 (the roadmap's slot count).
pub struct ModelSlotBank<const N: usize = SLOT_COUNT> {
    slots: [Option<PreparedModel>; N],
    /// Index the audio thread is currently rendering.
    active_index: usize,
    /// Requested next-active index, consumed by [`take_pending`].
    ///
    /// [`take_pending`]: Self::take_pending
    pending_index: Option<usize>,
}

impl<const N: usize> ModelSlotBank<N> {
    /// Bank capacity `N`.
    pub const CAPACITY: usize = N;

    /// Create an empty bank with slot 0 active and no activation pending.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    pub fn new() -> Self {
        assert!(N > 0, "ModelSlotBank needs at least one slot");
        Self {
            slots: array::from_fn(|_| None),
            active_index: 0,
            pending_index: None,
        }
    }

    /// Bank capacity `N`.
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Load `model` into `slot_index`.  Any model previously in the slot
    /// is dropped.
    ///
    /// **Not** real-time safe — dropping the previous model may release
    /// heap allocations.
    pub fn load(&mut self, slot_index: usize, model: PreparedModel) -> Result<(), SlotError> {
        self.check_index(slot_index)?;
        self.slots[slot_index] = Some(model);
        Ok(())
    }

    /// Clear `slot_index` — the slot is now empty (dry pass-through).
    ///
    /// **Not** real-time safe — drops the previous model.
    pub fn clear(&mut self, slot_index: usize) -> Result<(), SlotError> {
        self.check_index(slot_index)?;
        self.slots[slot_index] = None;
        Ok(())
    }

    /// Whether `slot_index` currently holds a model.
    pub fn is_loaded(&self, slot_index: usize) -> Result<bool, SlotError> {
        self.check_index(slot_index)?;
        Ok(self.slots[slot_index].is_some())
    }

    /// Ask that `slot_index` become the active slot at the next audio
    /// block boundary.  Empty slots are legal — see module docs.
    pub fn request_activation(&mut self, slot_index: usize) -> Result<(), SlotError> {
        self.check_index(slot_index)?;
        self.pending_index = Some(slot_index);
        Ok(())
    }

    /// Currently active slot index.
    #[inline]
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// The pending activation intent, if any.  Test / diagnostics helper.
    #[inline]
    pub fn pending_index(&self) -> Option<usize> {
        self.pending_index
    }

    /// Consume any pending activation intent and make it the active slot.
    ///
    /// Returns the new active index if a switch occurred, `None`
    /// otherwise.  Intended to be called by the audio thread at the top
    /// of every block; safe to call even when no intent is pending.
    ///
    /// **Real-time safe** — an `Option::take` and one `usize` write.
    #[inline]
    pub fn take_pending(&mut self) -> Option<usize> {
        let next = self.pending_index.take()?;
        self.active_index = next;
        Some(next)
    }

    /// Read-only view of the currently active model, if the slot is loaded.
    ///
    /// The audio thread rendering path uses this — but note that Phase 2
    /// will wrap the active-slot access in a `triple_buffer` / `ArcSwap`
    /// handoff, so this direct accessor is only used inside single-
    /// threaded tests.
    #[inline]
    pub fn active(&self) -> Option<&PreparedModel> {
        self.slots[self.active_index].as_ref()
    }

    /// Mutable view of the currently active model (single-threaded only).
    #[inline]
    pub fn active_mut(&mut self) -> Option<&mut PreparedModel> {
        self.slots[self.active_index].as_mut()
    }

    /// Range-check helper used by the fallible public methods.
    #[inline]
    fn check_index(&self, slot_index: usize) -> Result<(), SlotError> {
        if slot_index >= N {
            Err(SlotError::IndexOutOfRange {
                index: slot_index,
                capacity: N,
            })
        } else {
            Ok(())
        }
    }
}

impl<const N: usize> Default for ModelSlotBank<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lstm::LstmCell;
    use crate::prepared::LstmModel;

    /// Build a trivial LSTM-based `PreparedModel` for slot-population tests.
    fn make_trivial_model() -> PreparedModel {
        let hidden_size = 2;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size * input_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )
        .expect("LstmCell::new");
        PreparedModel::Lstm(LstmModel::new(vec![cell], vec![0.0f32; hidden_size], 0.0))
    }

    /// A fresh bank has all slots empty and slot 0 active with no pending
    /// activation.
    #[test]
    fn fresh_bank_is_empty_and_slot_zero_active() {
        let bank: ModelSlotBank = ModelSlotBank::new();
        assert_eq!(bank.active_index(), 0);
        assert!(bank.pending_index().is_none());
        assert!(bank.active().is_none());
        for slot_index in 0..SLOT_COUNT {
            assert_eq!(bank.is_loaded(slot_index), Ok(false));
        }
    }

    /// `load` populates the target slot and can be verified via `is_loaded`.
    #[test]
    fn load_populates_slot() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        bank.load(3, make_trivial_model()).expect("load");
        assert_eq!(bank.is_loaded(3), Ok(true));
        assert_eq!(bank.is_loaded(4), Ok(false));
    }

    /// `clear` removes the model from the slot.
    #[test]
    fn clear_empties_slot() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        bank.load(1, make_trivial_model()).expect("load");
        bank.clear(1).expect("clear");
        assert_eq!(bank.is_loaded(1), Ok(false));
    }

    /// Out-of-range indices are rejected by every fallible method.
    #[test]
    fn out_of_range_indices_are_rejected() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        let bad_index = SLOT_COUNT;
        assert_eq!(
            bank.load(bad_index, make_trivial_model()),
            Err(SlotError::IndexOutOfRange {
                index: bad_index,
                capacity: SLOT_COUNT,
            }),
        );
        assert!(bank.clear(bad_index).is_err());
        assert!(bank.request_activation(bad_index).is_err());
        assert!(bank.is_loaded(bad_index).is_err());
    }

    /// `request_activation` records the intent but does not change `active`.
    #[test]
    fn request_activation_records_intent() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        bank.load(2, make_trivial_model()).expect("load");
        bank.request_activation(2).expect("request");
        assert_eq!(bank.active_index(), 0);
        assert_eq!(bank.pending_index(), Some(2));
    }

    /// `take_pending` promotes the intent to `active` and clears it.
    #[test]
    fn take_pending_promotes_intent() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        bank.load(5, make_trivial_model()).expect("load");
        bank.request_activation(5).expect("request");
        assert_eq!(bank.take_pending(), Some(5));
        assert_eq!(bank.active_index(), 5);
        assert!(bank.active().is_some());
        // Second call finds nothing pending.
        assert_eq!(bank.take_pending(), None);
    }

    /// Activating an *empty* slot is legal and produces a dry pass-through.
    #[test]
    fn activating_empty_slot_is_legal() {
        let mut bank: ModelSlotBank = ModelSlotBank::new();
        bank.request_activation(4).expect("request");
        bank.take_pending();
        assert_eq!(bank.active_index(), 4);
        assert!(bank.active().is_none()); // dry pass-through
    }

    /// `capacity()` and `CAPACITY` agree.
    #[test]
    fn capacity_is_generic_parameter() {
        let bank: ModelSlotBank<3> = ModelSlotBank::new();
        assert_eq!(bank.capacity(), 3);
        assert_eq!(ModelSlotBank::<3>::CAPACITY, 3);
    }

    /// A zero-capacity bank panics on construction (invariant).
    #[test]
    #[should_panic(expected = "ModelSlotBank needs at least one slot")]
    fn zero_capacity_panics() {
        let _bank: ModelSlotBank<0> = ModelSlotBank::new();
    }
}
