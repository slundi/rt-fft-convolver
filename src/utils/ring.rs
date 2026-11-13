//! Fixed-capacity, real-time-safe FIFO ring buffer.
//!
//! A [`Ring<T>`] is a bounded first-in-first-out queue backed by a single
//! pre-allocated contiguous array.  It is the shared building block for any
//! effect that has to reconcile the host's arbitrary block size with an
//! internal fixed hop size — STFT analysis/synthesis framing, delay lines,
//! pre-delay, and look-ahead limiters all sit on top of it.
//!
//! # Real-time contract
//!
//! Every method other than the constructor is **allocation-free, lock-free
//! and panic-free**, so the whole hot path (`push` / `pop` / `push_slice` /
//! `pop_slice` / `as_slices`) is safe to call from `process()`.  All storage
//! is reserved once in [`Ring::with_capacity`], which is the only method that
//! allocates and therefore the only one that must run off the audio thread.
//!
//! # Why a `head + len` layout (not `head + tail`)
//!
//! A classic `head`/`tail` ring wastes one slot to disambiguate the empty and
//! full states (`head == tail` is ambiguous).  Storing `head` (index of the
//! oldest element) together with `len` (number of valid elements) removes that
//! ambiguity, so the full capacity is usable and `is_empty` / `is_full` are
//! trivial comparisons.  The write cursor is derived on demand as
//! `(head + len) mod capacity`.

/// Fixed-capacity FIFO ring buffer over `Copy` elements.
///
/// The `Default` bound lets the backing store be zero-filled at construction
/// so the hot path never touches uninitialised memory and stays `unsafe`-free.
/// For audio the element type is `f32`, which is trivially `Copy + Default`.
///
/// ```
/// use rt_fft_convolver::Ring;
///
/// // Sized once, off the audio thread.
/// let mut ring: Ring<f32> = Ring::with_capacity(4);
/// assert_eq!(ring.push_slice(&[1.0, 2.0, 3.0]), 3);
///
/// let mut frame = [0.0f32; 2];
/// assert_eq!(ring.pop_slice(&mut frame), 2);
/// assert_eq!(frame, [1.0, 2.0]);
/// assert_eq!(ring.len(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct Ring<T> {
    /// Backing storage, length == capacity.  Slots outside the valid region
    /// `[head, head + len)` (mod capacity) hold stale/default values and are
    /// never read.
    buffer: Vec<T>,

    /// Index of the oldest (front) element.  Always in `[0, capacity)`.
    head: usize,

    /// Number of valid elements currently queued.  Always in `[0, capacity]`.
    len: usize,
}

impl<T: Copy + Default> Ring<T> {
    /// Create an empty ring able to hold `capacity` elements.
    ///
    /// **Allocates** — call from a non-real-time thread (construction / preset
    /// change), never from `process()`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`; a zero-length FIFO can never accept an
    /// element and is always a construction-time bug.  The panic is confined
    /// to this off-RT constructor.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "Ring capacity must be non-zero");
        Self {
            buffer: vec![T::default(); capacity],
            head: 0,
            len: 0,
        }
    }

    /// Maximum number of elements the ring can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Number of elements currently queued.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of additional elements that can be pushed before the ring is
    /// full.
    #[inline]
    pub fn free(&self) -> usize {
        self.capacity() - self.len
    }

    /// `true` when no elements are queued.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` when the ring holds `capacity` elements and cannot accept more.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity()
    }

    /// Empty the ring in O(1).
    ///
    /// The backing store is left untouched (stale values remain) — validity is
    /// tracked solely by `len`, so nothing needs to be cleared.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Push one element onto the back.
    ///
    /// Returns `false` (and drops `value`) when the ring is already full,
    /// leaving the contents unchanged.  Callers on the audio thread size the
    /// ring so this never happens; the boolean lets non-RT callers detect
    /// overflow without a panic.
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        if self.is_full() {
            return false;
        }
        let write = self.write_index();
        self.buffer[write] = value;
        self.len += 1;
        true
    }

    /// Pop the oldest element from the front, or `None` when empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let value = self.buffer[self.head];
        self.head = self.wrap(self.head + 1);
        self.len -= 1;
        Some(value)
    }

    /// Borrow the oldest element without removing it, or `None` when empty.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self.buffer[self.head])
        }
    }

    /// Bulk-push from a slice; returns the number of elements actually written.
    ///
    /// Writes `min(src.len(), free())` elements — a short count signals the
    /// ring filled up.  The copy is done with up to two [`copy_from_slice`]
    /// calls (one before the wrap point, one after), which the compiler lowers
    /// to `memcpy`/SIMD rather than a per-element loop.
    ///
    /// [`copy_from_slice`]: slice::copy_from_slice
    pub fn push_slice(&mut self, src: &[T]) -> usize {
        let count = src.len().min(self.free());
        let capacity = self.capacity();
        let write = self.write_index();

        // First segment: from the write cursor up to (at most) the buffer end.
        let first = (capacity - write).min(count);
        self.buffer[write..write + first].copy_from_slice(&src[..first]);

        // Second segment: the remainder wraps back to index 0.
        let second = count - first;
        if second > 0 {
            self.buffer[..second].copy_from_slice(&src[first..first + second]);
        }

        self.len += count;
        count
    }

    /// Bulk-pop into a slice; returns the number of elements actually read.
    ///
    /// Reads `min(dst.len(), len())` elements from the front and advances the
    /// head.  Like [`push_slice`](Self::push_slice) the transfer uses up to two
    /// `copy_from_slice` calls around the wrap point.
    pub fn pop_slice(&mut self, dst: &mut [T]) -> usize {
        let count = dst.len().min(self.len);
        let capacity = self.capacity();

        // First segment: from the head up to (at most) the buffer end.
        let first = (capacity - self.head).min(count);
        dst[..first].copy_from_slice(&self.buffer[self.head..self.head + first]);

        // Second segment: the remainder wraps back to index 0.
        let second = count - first;
        if second > 0 {
            dst[first..first + second].copy_from_slice(&self.buffer[..second]);
        }

        self.head = self.wrap(self.head + count);
        self.len -= count;
        count
    }

    /// Return the queued elements, front-first, as up to two contiguous slices.
    ///
    /// Mirrors [`VecDeque::as_slices`]: the first slice always holds the older
    /// elements.  When the valid region does not straddle the wrap point the
    /// second slice is empty.  This is what an STFT framer uses to peek the
    /// analysis window without copying, deciding per call whether it can read a
    /// full contiguous frame or must stitch across the two segments.
    ///
    /// [`VecDeque::as_slices`]: std::collections::VecDeque::as_slices
    #[inline]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        if self.is_empty() {
            return (&[], &[]);
        }
        let capacity = self.capacity();
        let end = self.head + self.len;
        if end <= capacity {
            // Contiguous: the whole region fits before the wrap point.
            (&self.buffer[self.head..end], &[])
        } else {
            // Split: tail runs to the buffer end, remainder wraps to index 0.
            (
                &self.buffer[self.head..capacity],
                &self.buffer[..end - capacity],
            )
        }
    }

    /// Index at which the next pushed element will be written.
    #[inline]
    fn write_index(&self) -> usize {
        self.wrap(self.head + self.len)
    }

    /// Reduce an index that may have overrun by up to one lap back into
    /// `[0, capacity)`.  A subtraction suffices because callers only ever
    /// advance by at most `capacity` in a single step, so a `%` is avoided.
    #[inline]
    fn wrap(&self, index: usize) -> usize {
        let capacity = self.capacity();
        if index >= capacity {
            index - capacity
        } else {
            index
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh ring reports its capacity, empty state, and full free room.
    #[test]
    fn new_ring_is_empty() {
        let ring: Ring<f32> = Ring::with_capacity(4);
        assert_eq!(ring.capacity(), 4);
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.free(), 4);
        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.peek(), None);
    }

    /// A zero capacity is a construction-time bug and must panic.
    #[test]
    #[should_panic(expected = "capacity must be non-zero")]
    fn zero_capacity_panics() {
        let _ring: Ring<f32> = Ring::with_capacity(0);
    }

    /// Single push/pop preserves FIFO order.
    #[test]
    fn push_then_pop_is_fifo() {
        let mut ring: Ring<i32> = Ring::with_capacity(3);
        assert!(ring.push(10));
        assert!(ring.push(20));
        assert!(ring.push(30));
        assert!(ring.is_full());
        assert_eq!(ring.pop(), Some(10));
        assert_eq!(ring.pop(), Some(20));
        assert_eq!(ring.pop(), Some(30));
        assert_eq!(ring.pop(), None);
    }

    /// `push` refuses to overwrite when full and leaves contents intact.
    #[test]
    fn push_on_full_ring_is_rejected() {
        let mut ring: Ring<i32> = Ring::with_capacity(2);
        assert!(ring.push(1));
        assert!(ring.push(2));
        assert!(!ring.push(3)); // rejected
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop(), Some(1));
        assert_eq!(ring.pop(), Some(2));
    }

    /// `peek` returns the front element without consuming it.
    #[test]
    fn peek_does_not_consume() {
        let mut ring: Ring<i32> = Ring::with_capacity(2);
        ring.push(7);
        assert_eq!(ring.peek(), Some(&7));
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.pop(), Some(7));
    }

    /// Interleaved push/pop forces the cursors past the wrap point and must
    /// still return elements in FIFO order.
    #[test]
    fn wrap_around_preserves_order() {
        let mut ring: Ring<i32> = Ring::with_capacity(3);
        ring.push(1);
        ring.push(2);
        assert_eq!(ring.pop(), Some(1)); // head advances to 1
        ring.push(3);
        ring.push(4); // write wraps to index 0
        assert!(ring.is_full());
        assert_eq!(ring.pop(), Some(2));
        assert_eq!(ring.pop(), Some(3));
        assert_eq!(ring.pop(), Some(4));
        assert!(ring.is_empty());
    }

    /// `push_slice` returns the count written and truncates at capacity.
    #[test]
    fn push_slice_truncates_at_capacity() {
        let mut ring: Ring<f32> = Ring::with_capacity(4);
        let written = ring.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(written, 4);
        assert!(ring.is_full());
    }

    /// `pop_slice` returns the count read and never over-reads.
    #[test]
    fn pop_slice_reads_available() {
        let mut ring: Ring<f32> = Ring::with_capacity(4);
        ring.push_slice(&[1.0, 2.0, 3.0]);
        let mut out = [0.0f32; 5];
        let read = ring.pop_slice(&mut out);
        assert_eq!(read, 3);
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
        assert!(ring.is_empty());
    }

    /// A bulk push/pop that straddles the wrap point must round-trip intact —
    /// this exercises the two-segment `copy_from_slice` paths.
    #[test]
    fn slice_roundtrip_across_wrap() {
        let mut ring: Ring<f32> = Ring::with_capacity(4);
        // Advance the head so subsequent writes wrap.
        ring.push_slice(&[9.0, 9.0, 9.0]);
        let mut drain = [0.0f32; 3];
        ring.pop_slice(&mut drain); // head now at index 3

        let source = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(ring.push_slice(&source), 4); // writes at 3,0,1,2 (wraps)
        assert!(ring.is_full());

        let mut out = [0.0f32; 4];
        assert_eq!(ring.pop_slice(&mut out), 4);
        assert_eq!(out, source);
    }

    /// `as_slices` yields one slice when the region is contiguous and two when
    /// it straddles the wrap point, always front-first.
    #[test]
    fn as_slices_reflects_wrap() {
        let mut ring: Ring<i32> = Ring::with_capacity(4);
        ring.push_slice(&[1, 2, 3]);
        let (front, back) = ring.as_slices();
        assert_eq!(front, &[1, 2, 3]);
        assert!(back.is_empty());

        // Force a wrap: drop two from the front, push three more.
        ring.pop();
        ring.pop();
        ring.push_slice(&[4, 5]); // valid region: indices 2,3,0 -> [3,4,5]
        let (front, back) = ring.as_slices();
        // Reassembling the two segments must give FIFO order.
        let joined: Vec<i32> = front.iter().chain(back.iter()).copied().collect();
        assert_eq!(joined, vec![3, 4, 5]);
        assert!(!back.is_empty(), "region should straddle the wrap point");
    }

    /// `clear` resets length in O(1) and makes the ring reusable.
    #[test]
    fn clear_resets_state() {
        let mut ring: Ring<i32> = Ring::with_capacity(3);
        ring.push_slice(&[1, 2, 3]);
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.pop(), None);
        assert!(ring.push(42));
        assert_eq!(ring.pop(), Some(42));
    }

    /// Pushing/popping empty slices are well-defined no-ops.
    #[test]
    fn empty_slice_transfers_are_noops() {
        let mut ring: Ring<f32> = Ring::with_capacity(2);
        assert_eq!(ring.push_slice(&[]), 0);
        let mut empty: [f32; 0] = [];
        assert_eq!(ring.pop_slice(&mut empty), 0);
        assert!(ring.is_empty());
    }
}
