//! Lock-free, wait-free handoff of the active convolver from the control
//! thread to the audio thread.
//!
//! # Why not a mutex?
//!
//! The audio callback must never block.  A mutex held by the worker thread
//! while it swaps in a freshly-built convolver could stall the callback and
//! cause a dropout.  Instead we use a [`triple_buffer`], which lets the
//! producer publish a new value and the consumer read the latest one without
//! either ever waiting on the other.
//!
//! # RT-safety of the audio side
//!
//! [`IrConsumer::poll`] is **wait-free and allocation-free**: it swaps a pair
//! of buffer indices and hands back a mutable reference to the reader's own
//! buffer.  Crucially, the convolver the audio thread stops using is *not*
//! dropped on the audio thread — `triple_buffer` recycles it into the buffer
//! pool and it is eventually overwritten (and dropped) by the **producer**.
//! No deallocation ever happens on the audio thread.
//!
//! # Producer side
//!
//! [`IrPublisher::publish`] moves a prepared convolver into the buffer.  It is
//! meant for the control / worker thread and may drop the previously-published
//! value (deallocating), so it must not run on the audio thread.
//!
//! ```no_run
//! use rt_ir_plugin_core::ir_handoff;
//!
//! // `T` is your convolver type; here we illustrate with a Vec.
//! let (mut publisher, mut consumer) = ir_handoff::<Vec<f32>>();
//!
//! // control thread:
//! publisher.publish(vec![1.0, 0.0, 0.0]);
//!
//! // audio thread, each block:
//! if let Some(_conv) = consumer.poll() {
//!     // run `_conv` for this block
//! }
//! ```

use triple_buffer::{Input, Output, TripleBuffer};

/// Producer half of the handoff — lives on the control / worker thread.
pub struct IrPublisher<T: Send> {
    input: Input<Option<T>>,
}

/// Consumer half of the handoff — lives on the audio thread.
pub struct IrConsumer<T: Send> {
    output: Output<Option<T>>,
}

/// Create a connected [`IrPublisher`] / [`IrConsumer`] pair, initially empty.
///
/// Until the first [`IrPublisher::publish`], the consumer's [`poll`] and
/// [`current`] return `None`.
///
/// [`poll`]: IrConsumer::poll
/// [`current`]: IrConsumer::current
pub fn ir_handoff<T: Send>() -> (IrPublisher<T>, IrConsumer<T>) {
    // `Option<T>: Default` for every `T` (it is `None`), so the triple buffer
    // starts empty without requiring `T: Default`.
    let (input, output) = TripleBuffer::<Option<T>>::default().split();
    (IrPublisher { input }, IrConsumer { output })
}

impl<T: Send> IrPublisher<T> {
    /// Publish a new convolver, making it the value the audio thread will pick
    /// up on its next [`IrConsumer::poll`].
    ///
    /// Overwrites any previously-published-but-unconsumed value.  **May drop
    /// the displaced value (deallocating) — do not call on the audio thread.**
    pub fn publish(&mut self, value: T) {
        self.input.write(Some(value));
    }
}

impl<T: Send> IrConsumer<T> {
    /// Fetch the latest published convolver and report whether it changed
    /// since the previous call.
    ///
    /// Returns `(value, changed)` where `changed` is `true` only when the
    /// producer published something new since the last poll.  **Wait-free and
    /// allocation-free** — safe for the audio thread.
    pub fn poll_changed(&mut self) -> (Option<&mut T>, bool) {
        let changed = self.output.update();
        (self.output.output_buffer_mut().as_mut(), changed)
    }

    /// Fetch the latest published convolver.
    ///
    /// Equivalent to [`poll_changed`] discarding the change flag.  **Wait-free
    /// and allocation-free** — safe for the audio thread.
    ///
    /// [`poll_changed`]: IrConsumer::poll_changed
    #[inline]
    pub fn poll(&mut self) -> Option<&mut T> {
        self.output.update();
        self.output.output_buffer_mut().as_mut()
    }

    /// Borrow the current convolver **without** checking for an update.
    ///
    /// Cheaper than [`poll`] when you have already polled this block.
    ///
    /// [`poll`]: IrConsumer::poll
    #[inline]
    pub fn current(&mut self) -> Option<&mut T> {
        self.output.output_buffer_mut().as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_until_first_publish() {
        let (_publisher, mut consumer) = ir_handoff::<u32>();
        assert_eq!(consumer.poll().copied(), None);
        let (value, changed) = consumer.poll_changed();
        assert_eq!(value.copied(), None);
        assert!(!changed);
    }

    #[test]
    fn consumer_sees_published_value() {
        let (mut publisher, mut consumer) = ir_handoff::<u32>();
        publisher.publish(42);

        let (value, changed) = consumer.poll_changed();
        assert_eq!(value.copied(), Some(42));
        assert!(changed, "first publish must register as a change");

        // No new publish → no change, same value.
        let (value, changed) = consumer.poll_changed();
        assert_eq!(value.copied(), Some(42));
        assert!(!changed);
    }

    #[test]
    fn latest_publish_wins() {
        let (mut publisher, mut consumer) = ir_handoff::<u32>();
        publisher.publish(1);
        publisher.publish(2);
        publisher.publish(3);
        // The consumer only ever observes the freshest value.
        assert_eq!(consumer.poll().copied(), Some(3));
    }

    #[test]
    fn audio_thread_can_mutate_in_place() {
        let (mut publisher, mut consumer) = ir_handoff::<Vec<f32>>();
        publisher.publish(vec![0.0; 4]);
        // Simulate the audio thread writing into the active convolver's state.
        if let Some(buf) = consumer.poll() {
            buf[0] = 1.0;
        }
        // The mutation is visible on the reader's buffer until the next update.
        assert_eq!(consumer.current().map(|b| b[0]), Some(1.0));
    }

    #[test]
    fn crosses_threads() {
        let (mut publisher, mut consumer) = ir_handoff::<u64>();
        let handle = std::thread::spawn(move || {
            publisher.publish(7);
            publisher // return so it is not dropped mid-test
        });
        let _publisher = handle.join().expect("producer thread");

        // Spin until the published value propagates (bounded: it is already
        // written before the thread joined).
        let mut seen = None;
        for _ in 0..1_000 {
            if let Some(&mut v) = consumer.poll() {
                seen = Some(v);
                break;
            }
        }
        assert_eq!(seen, Some(7));
    }
}
