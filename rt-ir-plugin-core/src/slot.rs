//! [`IrSlotBank`] — a fixed bank of `N` IR slots with an "activation intent".
//!
//! # Model
//!
//! A plugin owns `N` (default 8) slots.  Each slot is either empty or holds a
//! prepared payload `T` (typically a built convolver, or a [`PreparedIr`]).
//! Loading happens off the audio thread; the audio side only ever needs the
//! *active* slot index, so activation is expressed as an **intent** that the
//! audio thread consumes at a block boundary.
//!
//! ```text
//! control thread:  load(3, ir)          request_activation(3)
//!                       │                        │
//!                       ▼                        ▼
//!                 slots[3] = Some(ir)      pending = Some(3)
//!                                                 │  (next block)
//! audio thread:                     take_pending() ─┴─► active = 3
//! ```
//!
//! The bank itself performs **no** cross-thread synchronisation — that is the
//! job of the lock-free [`handoff`].  It is a plain state container, which
//! keeps it trivially testable and lets the plugin decide how the two threads
//! share it.
//!
//! # Empty slots are legal
//!
//! Per the roadmap, "empty slots pass audio through dry — never panic".
//! [`request_activation`] therefore accepts *any* in-range index, loaded or
//! not; the plugin plays dry when the active slot is empty.  Only an
//! out-of-range index is an error.
//!
//! [`PreparedIr`]: crate::ir::PreparedIr
//! [`handoff`]: crate::handoff
//! [`request_activation`]: IrSlotBank::request_activation

use std::array;

/// Error returned by fallible [`IrSlotBank`] operations.
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotError::IndexOutOfRange { index, capacity } => {
                write!(f, "slot index {index} out of range (capacity {capacity})")
            }
        }
    }
}

impl std::error::Error for SlotError {}

/// A bank of `N` IR slots plus a single-slot "activate on next block" intent.
///
/// `T` is whatever the plugin wants to keep per slot — most often a built
/// convolver, but [`PreparedIr`](crate::ir::PreparedIr) or a file path work
/// too.  `N` defaults to 8 (the roadmap's slot count).
pub struct IrSlotBank<T, const N: usize = 8> {
    slots: [Option<T>; N],
    /// Index the audio thread is currently rendering.
    active: usize,
    /// Requested next-active index, consumed by [`take_pending`].
    ///
    /// [`take_pending`]: IrSlotBank::take_pending
    pending: Option<usize>,
}

impl<T, const N: usize> IrSlotBank<T, N> {
    /// Number of slots in the bank.
    pub const CAPACITY: usize = N;

    /// Create an empty bank with slot 0 active and no activation pending.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0` (a bank must have at least one slot).
    pub fn new() -> Self {
        assert!(N > 0, "IrSlotBank needs at least one slot");
        Self {
            slots: array::from_fn(|_| None),
            active: 0,
            pending: None,
        }
    }

    /// Bank capacity `N`.
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Store `value` in slot `index`, returning the previous occupant.
    ///
    /// **May drop the previous payload** (deallocating) — call off the audio
    /// thread.
    ///
    /// # Errors
    ///
    /// [`SlotError::IndexOutOfRange`] if `index >= N`.
    pub fn load(&mut self, index: usize, value: T) -> Result<Option<T>, SlotError> {
        let slot = self.slot_mut(index)?;
        Ok(slot.replace(value))
    }

    /// Empty slot `index`, returning whatever it held.
    ///
    /// # Errors
    ///
    /// [`SlotError::IndexOutOfRange`] if `index >= N`.
    pub fn clear(&mut self, index: usize) -> Result<Option<T>, SlotError> {
        let slot = self.slot_mut(index)?;
        Ok(slot.take())
    }

    /// Borrow the payload in slot `index`, or `None` if empty / out of range.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index).and_then(Option::as_ref)
    }

    /// Whether slot `index` currently holds a payload.
    #[inline]
    pub fn is_loaded(&self, index: usize) -> bool {
        self.get(index).is_some()
    }

    /// Number of non-empty slots.
    #[inline]
    pub fn loaded_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Index the bank considers active (last consumed via [`take_pending`]).
    ///
    /// [`take_pending`]: IrSlotBank::take_pending
    #[inline]
    pub fn active_index(&self) -> usize {
        self.active
    }

    /// Borrow the active slot's payload, or `None` if that slot is empty.
    #[inline]
    pub fn active(&self) -> Option<&T> {
        self.get(self.active)
    }

    /// Request that slot `index` become active on the next block.
    ///
    /// Accepts empty slots (the plugin then plays dry); only an out-of-range
    /// index is rejected.  A later request overwrites an unconsumed one.
    ///
    /// # Errors
    ///
    /// [`SlotError::IndexOutOfRange`] if `index >= N`.
    pub fn request_activation(&mut self, index: usize) -> Result<(), SlotError> {
        if index >= N {
            return Err(SlotError::IndexOutOfRange { index, capacity: N });
        }
        self.pending = Some(index);
        Ok(())
    }

    /// Whether an activation request is waiting to be consumed.
    #[inline]
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Consume a pending activation, if any, updating [`active_index`].
    ///
    /// Returns the newly-active index, or `None` when nothing was requested
    /// (or the request pointed at the already-active slot). **Allocation-free
    /// — safe to call from the audio thread** at a block boundary.
    ///
    /// [`active_index`]: IrSlotBank::active_index
    pub fn take_pending(&mut self) -> Option<usize> {
        match self.pending.take() {
            Some(index) if index != self.active => {
                self.active = index;
                Some(index)
            }
            // A request for the already-active slot is a no-op switch.
            _ => None,
        }
    }

    /// Bounds-checked mutable slot access.
    fn slot_mut(&mut self, index: usize) -> Result<&mut Option<T>, SlotError> {
        let capacity = N;
        self.slots
            .get_mut(index)
            .ok_or(SlotError::IndexOutOfRange { index, capacity })
    }
}

impl<T, const N: usize> Default for IrSlotBank<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Payloads are plain values here — the bank is generic and does not care.
    type Bank = IrSlotBank<&'static str, 8>;

    #[test]
    fn new_bank_is_empty_and_defaults_to_slot_zero() {
        let bank = Bank::new();
        assert_eq!(bank.capacity(), 8);
        assert_eq!(bank.loaded_count(), 0);
        assert_eq!(bank.active_index(), 0);
        assert!(!bank.has_pending());
        assert!(bank.active().is_none());
    }

    #[test]
    fn load_stores_and_returns_previous() {
        let mut bank = Bank::new();
        assert_eq!(bank.load(2, "cab-a"), Ok(None));
        assert!(bank.is_loaded(2));
        assert_eq!(bank.get(2), Some(&"cab-a"));
        assert_eq!(bank.loaded_count(), 1);
        // Overwriting hands back the previous occupant.
        assert_eq!(bank.load(2, "cab-b"), Ok(Some("cab-a")));
        assert_eq!(bank.get(2), Some(&"cab-b"));
        assert_eq!(bank.loaded_count(), 1);
    }

    #[test]
    fn clear_empties_the_slot() {
        let mut bank = Bank::new();
        bank.load(0, "x").unwrap();
        assert_eq!(bank.clear(0), Ok(Some("x")));
        assert!(!bank.is_loaded(0));
        assert_eq!(bank.clear(0), Ok(None));
    }

    #[test]
    fn out_of_range_index_errors() {
        let mut bank = Bank::new();
        let err = Err(SlotError::IndexOutOfRange {
            index: 8,
            capacity: 8,
        });
        assert_eq!(bank.load(8, "x"), err);
        assert_eq!(bank.clear(8), err);
        assert_eq!(
            bank.request_activation(8),
            Err(SlotError::IndexOutOfRange {
                index: 8,
                capacity: 8
            })
        );
        assert_eq!(bank.get(8), None);
        assert!(!bank.is_loaded(8));
    }

    #[test]
    fn activation_intent_is_consumed_once() {
        let mut bank = Bank::new();
        bank.load(3, "cab").unwrap();
        bank.request_activation(3).unwrap();
        assert!(bank.has_pending());

        // Consuming updates the active index and clears the intent.
        assert_eq!(bank.take_pending(), Some(3));
        assert_eq!(bank.active_index(), 3);
        assert!(!bank.has_pending());
        // Nothing left to consume.
        assert_eq!(bank.take_pending(), None);
    }

    #[test]
    fn requesting_the_active_slot_is_a_noop() {
        let mut bank = Bank::new();
        // Slot 0 is already active.
        bank.request_activation(0).unwrap();
        assert_eq!(bank.take_pending(), None);
        assert_eq!(bank.active_index(), 0);
    }

    #[test]
    fn empty_slots_are_activatable() {
        // "Empty slots pass audio through dry — never panic": activating an
        // unloaded slot is allowed; the plugin renders dry.
        let mut bank = Bank::new();
        bank.request_activation(5).unwrap();
        assert_eq!(bank.take_pending(), Some(5));
        assert_eq!(bank.active_index(), 5);
        assert!(bank.active().is_none());
    }

    #[test]
    fn latest_request_wins_before_consumption() {
        let mut bank = Bank::new();
        bank.request_activation(1).unwrap();
        bank.request_activation(4).unwrap();
        assert_eq!(bank.take_pending(), Some(4));
        assert_eq!(bank.active_index(), 4);
    }

    #[test]
    fn const_generic_capacity_is_respected() {
        let bank: IrSlotBank<u8, 4> = IrSlotBank::new();
        assert_eq!(bank.capacity(), 4);
        assert_eq!(IrSlotBank::<u8, 4>::CAPACITY, 4);
    }
}
