//! MIDI → slot-index routing.
//!
//! Both plugins let the guitarist switch IR slots live from a MIDI foot
//! controller.  Two message types map to a slot (roadmap Phase 0):
//!
//! - **Program Change** — program number *n* selects slot *n* (0-based).  A
//!   program beyond the slot count is ignored.
//! - **Control Change** — a *user-assignable* CC number; its value is divided
//!   by 16 to pick a slot (`0-15 → 0`, `16-31 → 1`, …, `112-127 → 7`), which
//!   maps the full 0-127 range onto 8 slots.
//!
//! These are the framework-agnostic *policy* functions; the plugin feeds them
//! events decoded from whatever its host delivers (e.g. `nih_plug`'s
//! `NoteEvent`).  Host-UI parameter changes route through the same
//! [`IrSlotBank::request_activation`](crate::slot::IrSlotBank::request_activation)
//! call the router feeds, so all three sources share one activation path.

/// A MIDI event relevant to slot switching, already decoded from raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotEvent {
    /// MIDI Program Change with the given program number (0-127).
    ProgramChange(u8),
    /// MIDI Control Change: controller number and its value (each 0-127).
    ControlChange {
        /// Controller (CC) number.
        controller: u8,
        /// Controller value.
        value: u8,
    },
}

/// Map a Program-Change number to a slot index for a bank of `num_slots`.
///
/// Returns `None` when the program number addresses a slot that does not
/// exist, so the caller can ignore it rather than clamp to a wrong slot.
#[inline]
pub fn program_to_slot(program: u8, num_slots: usize) -> Option<usize> {
    let index = program as usize;
    (index < num_slots).then_some(index)
}

/// Map a CC value (0-127) to a slot index for a bank of `num_slots`.
///
/// Divides the value by 16 (so 0-127 spans 8 buckets) and clamps to the last
/// slot, guaranteeing a valid index for any `num_slots >= 1`.
///
/// # Panics
///
/// Panics if `num_slots == 0`.
#[inline]
pub fn cc_to_slot(value: u8, num_slots: usize) -> usize {
    assert!(num_slots > 0, "num_slots must be >= 1");
    ((value / 16) as usize).min(num_slots - 1)
}

/// Routes MIDI events to slot indices for a bank of a fixed size.
///
/// Holds the *assignable* CC number used for slot selection (set via
/// [`set_cc`], e.g. from a "MIDI learn" gesture).  Program Change is always
/// routed; Control Change is routed only when it matches the learned CC.
///
/// [`set_cc`]: MidiRouter::set_cc
#[derive(Debug, Clone)]
pub struct MidiRouter {
    num_slots: usize,
    /// The CC number bound to slot selection, or `None` if unassigned.
    cc: Option<u8>,
}

impl MidiRouter {
    /// Create a router for a bank of `num_slots`, with no CC assigned yet.
    ///
    /// # Panics
    ///
    /// Panics if `num_slots == 0`.
    pub fn new(num_slots: usize) -> Self {
        assert!(num_slots > 0, "num_slots must be >= 1");
        Self {
            num_slots,
            cc: None,
        }
    }

    /// Bind (or unbind, with `None`) the CC number used for slot selection.
    #[inline]
    pub fn set_cc(&mut self, cc: Option<u8>) {
        self.cc = cc;
    }

    /// The currently-bound slot-selection CC number, if any.
    #[inline]
    pub fn cc(&self) -> Option<u8> {
        self.cc
    }

    /// Number of slots this router maps onto.
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Route an event to a slot index, or `None` if it selects no slot.
    ///
    /// - Program Change → [`program_to_slot`] (ignored if out of range).
    /// - Control Change → [`cc_to_slot`], but only when the controller matches
    ///   the assigned CC; other controllers return `None`.
    pub fn route(&self, event: SlotEvent) -> Option<usize> {
        match event {
            SlotEvent::ProgramChange(program) => program_to_slot(program, self.num_slots),
            SlotEvent::ControlChange { controller, value } => {
                (Some(controller) == self.cc).then(|| cc_to_slot(value, self.num_slots))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_maps_one_to_one_within_range() {
        assert_eq!(program_to_slot(0, 8), Some(0));
        assert_eq!(program_to_slot(7, 8), Some(7));
        // Out of range → ignored.
        assert_eq!(program_to_slot(8, 8), None);
        assert_eq!(program_to_slot(127, 8), None);
    }

    #[test]
    fn cc_value_buckets_into_eight_slots() {
        assert_eq!(cc_to_slot(0, 8), 0);
        assert_eq!(cc_to_slot(15, 8), 0);
        assert_eq!(cc_to_slot(16, 8), 1);
        assert_eq!(cc_to_slot(112, 8), 7);
        assert_eq!(cc_to_slot(127, 8), 7);
    }

    #[test]
    fn cc_clamps_for_small_banks() {
        // Four slots: value/16 can reach 7, must clamp to 3.
        assert_eq!(cc_to_slot(127, 4), 3);
        assert_eq!(cc_to_slot(0, 1), 0);
        assert_eq!(cc_to_slot(127, 1), 0);
    }

    #[test]
    fn router_routes_program_change_always() {
        let router = MidiRouter::new(8);
        assert_eq!(router.route(SlotEvent::ProgramChange(3)), Some(3));
        assert_eq!(router.route(SlotEvent::ProgramChange(99)), None);
    }

    #[test]
    fn router_routes_cc_only_when_assigned() {
        let mut router = MidiRouter::new(8);
        // No CC assigned yet → CC events are ignored.
        assert_eq!(
            router.route(SlotEvent::ControlChange {
                controller: 20,
                value: 100
            }),
            None
        );

        router.set_cc(Some(20));
        assert_eq!(router.cc(), Some(20));
        assert_eq!(
            router.route(SlotEvent::ControlChange {
                controller: 20,
                value: 0
            }),
            Some(0)
        );
        assert_eq!(
            router.route(SlotEvent::ControlChange {
                controller: 20,
                value: 127
            }),
            Some(7)
        );
        // A different controller is not the learned one.
        assert_eq!(
            router.route(SlotEvent::ControlChange {
                controller: 21,
                value: 127
            }),
            None
        );
    }

    #[test]
    fn router_can_unbind_cc() {
        let mut router = MidiRouter::new(8);
        router.set_cc(Some(64));
        router.set_cc(None);
        assert_eq!(
            router.route(SlotEvent::ControlChange {
                controller: 64,
                value: 64
            }),
            None
        );
    }
}
