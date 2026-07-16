//! [`WindowMax`] — sliding-window maximum of the true-peak envelope.
//!
//! # Why the limiter needs it
//!
//! The gain the limiter applies to the sample currently leaving the delay
//! line must be small enough for **that sample and every sample within the
//! look-ahead window ahead of it** — otherwise a peak arriving a few samples
//! later would clip.  So the gain computer is driven not by the instantaneous
//! peak but by the *maximum* peak over the window `[n − W + 1, n]` of the peak
//! envelope, where the delay-line alignment makes that window cover exactly
//! "this output sample plus its look-ahead future".  Taking the window maximum
//! is what makes the gain reduction *anticipate* the peak (the attack), and it
//! is what makes the brick-wall guarantee provable: the applied gain never
//! exceeds `ceiling / peak` for any in-flight sample.
//!
//! # Algorithm — monotonic deque
//!
//! A naïve max-over-window is `O(W)` per sample (far too slow: `W` can be
//! ~2000 samples at 96 kHz / 20 ms).  Instead we keep a **monotonically
//! decreasing deque** of candidate `(index, value)` pairs:
//!
//! - On push, drop every trailing candidate `≤` the new value (they can never
//!   be the maximum while the new, later, larger value is in the window).
//! - Drop the leading candidate once its index falls outside the window.
//! - The front is always the current maximum.
//!
//! Each value is pushed and popped at most once, so the amortised cost is
//! `O(1)` per sample.  The deque lives in a fixed ring buffer sized at
//! construction, so it never allocates on the audio thread.

/// A fixed-capacity sliding-window maximum over a stream of `f32` peaks.
pub struct WindowMax {
    /// Ring of candidate stream indices (monotonic with `values`).
    indices: Vec<u64>,
    /// Ring of candidate values, monotonically decreasing front→back.
    values: Vec<f32>,
    /// Front of the deque (index of the current maximum) in `[0, cap)`.
    head: usize,
    /// One past the back of the deque, in `[0, cap)`.
    tail: usize,
    /// Ring capacity (`max_window + 2`).
    cap: usize,
    /// Current window length in samples, in `1..=max_window`.
    window: u64,
    /// Running count of samples pushed — the stream index of the next push.
    counter: u64,
}

impl WindowMax {
    /// Create a sliding-window maximum sized for up to `max_window` samples,
    /// with the window initialised to `window`.
    ///
    /// **Allocates** — call off the audio thread.
    ///
    /// # Panics
    ///
    /// Panics if `window == 0` or `window > max_window`.
    pub fn new(max_window: usize, window: usize) -> Self {
        assert!(window >= 1, "window must be ≥ 1");
        assert!(window <= max_window, "window exceeds max_window");
        // At most `window` candidates can coexist plus the one being pushed
        // before the front is evicted, so `max_window + 2` slots always fit.
        let cap = max_window + 2;
        Self {
            indices: vec![0; cap],
            values: vec![0.0; cap],
            head: 0,
            tail: 0,
            cap,
            window: window as u64,
            counter: 0,
        }
    }

    /// The current window length in samples.
    #[inline]
    pub fn window(&self) -> usize {
        self.window as usize
    }

    /// The maximum window length this instance was sized for.
    #[inline]
    pub fn max_window(&self) -> usize {
        self.cap - 2
    }

    /// Change the window length (clamped to `1..=max_window`).
    ///
    /// Takes effect as the deque rolls forward; existing candidates outside
    /// the new (shorter) window are evicted on the next [`push`](Self::push).
    #[inline]
    pub fn set_window(&mut self, window: usize) {
        self.window = (window.max(1).min(self.max_window())) as u64;
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// Push one peak value and return the maximum over the current window
    /// (inclusive of this sample).
    ///
    /// **Real-time safe** — amortised `O(1)`, no allocation.
    #[inline]
    pub fn push(&mut self, value: f32) -> f32 {
        let index = self.counter;
        self.counter += 1;

        // Drop trailing candidates that this value dominates: a later,
        // larger-or-equal value shadows them for the rest of their lifetime.
        while !self.is_empty() {
            let back = (self.tail + self.cap - 1) % self.cap;
            if self.values[back] <= value {
                self.tail = back;
            } else {
                break;
            }
        }
        // Append the new candidate at the back.
        self.indices[self.tail] = index;
        self.values[self.tail] = value;
        self.tail = (self.tail + 1) % self.cap;

        // Evict the front while it has fallen out of the window
        // `[index − window + 1, index]`, i.e. front.index + window ≤ index.
        while self.indices[self.head] + self.window <= index {
            self.head = (self.head + 1) % self.cap;
        }

        self.values[self.head]
    }

    /// Clear all buffered candidates and rewind the stream counter.
    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.counter = 0;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: max over the last `window` samples.
    fn reference_max(data: &[f32], window: usize, upto: usize) -> f32 {
        let start = (upto + 1).saturating_sub(window);
        data[start..=upto].iter().copied().fold(f32::MIN, f32::max)
    }

    /// The deque matches a brute-force sliding maximum on a varied stream.
    #[test]
    fn matches_brute_force() {
        let data = [
            0.1, 0.9, 0.3, 0.3, 0.8, 0.8, 0.2, 1.0, 0.05, 0.05, 0.5, 0.7, 0.1, 0.1,
        ];
        let window = 4;
        let mut win = WindowMax::new(window, window);
        for (position, &value) in data.iter().enumerate() {
            let got = win.push(value);
            let expected = reference_max(&data, window, position);
            assert!(
                (got - expected).abs() < 1e-6,
                "at {position}: got {got}, expected {expected}"
            );
        }
    }

    /// Window 1 degenerates to the identity (each sample is its own max).
    #[test]
    fn window_one_is_identity() {
        let mut win = WindowMax::new(8, 1);
        for value in [0.3f32, 0.9, 0.1, 0.7] {
            assert_eq!(win.push(value), value);
        }
    }

    /// A single tall spike is held for exactly `window` samples then drops.
    #[test]
    fn spike_is_held_for_window_samples() {
        let window = 3;
        let mut win = WindowMax::new(window, window);
        assert_eq!(win.push(1.0), 1.0); // spike enters
        assert_eq!(win.push(0.0), 1.0); // still in window
        assert_eq!(win.push(0.0), 1.0); // last sample of the window
        assert_eq!(win.push(0.0), 0.0); // spike has fallen out
    }

    /// `set_window` shortens the effective window on subsequent pushes.
    #[test]
    fn set_window_shrinks() {
        let mut win = WindowMax::new(8, 8);
        win.push(1.0);
        win.set_window(2);
        win.push(0.0);
        // With window 2 the old spike (3 samples back) is out of range.
        assert_eq!(win.push(0.0), 0.0);
    }

    /// `reset` clears the held maximum.
    #[test]
    fn reset_clears() {
        let mut win = WindowMax::new(4, 4);
        win.push(1.0);
        win.reset();
        assert_eq!(win.push(0.2), 0.2);
    }
}
