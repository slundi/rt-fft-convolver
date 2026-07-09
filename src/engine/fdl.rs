//! Frequency-domain delay line (FDL) for partitioned convolution.
//!
//! An FDL is a fixed-depth circular buffer of complex frequency-domain spectra.
//! At each audio block the newest input spectrum is pushed in via [`push`]; the
//! [`accumulate`] method then dot-products all stored spectra against the
//! corresponding IR partition spectra.
//!
//! # Role in uniform partitioned convolution
//!
//! The frequency-domain output at block `t` is:
//!
//! ```text
//! Y[t] = Σ_{k=0}^{P-1}  X[t-k]  ⊙  H[k]
//! ```
//!
//! where `X[t-k]` is the real-input 2N-point FFT of the input from `k`
//! blocks ago (stored as its `N + 1` unique complex bins — the rest are
//! conjugate-symmetric mirrors) and `H[k]` is the pre-computed FFT of the
//! k-th IR partition.  The FDL stores the P most recent `X` values and
//! [`accumulate`] evaluates the sum above in O(P · N) complex
//! multiplications.
//!
//! # Memory layout
//!
//! Internally the buffer is a flat circular array of `P` slots, each holding
//! `spectrum_size` complex bins.  The write pointer advances by one slot per
//! [`push`]; no memory is ever moved.
//!
//! ```text
//!  P = 3 slots, write_pos after three pushes (A, B, C):
//!
//!  slot index : 0    1    2
//!  content    : A    B    C
//!  write_pos  : 0   (will overwrite A on the next push)
//!
//!  delay 0 (most recent) → slot (write_pos + P − 1) % P = 2  → C
//!  delay 1               → slot (write_pos + P − 2) % P = 1  → B
//!  delay 2               → slot (write_pos + P − 3) % P = 0  → A
//! ```
//!
//! [`push`]: FreqDomainDelayLine::push
//! [`accumulate`]: FreqDomainDelayLine::accumulate

use rustfft::num_complex::Complex;
use wide::f32x8;

/// Multiply-accumulate `acc[i] += slot[i] * ir[i]` for complex f32 slices.
///
/// Uses `f32x8` to process **4 complex numbers per SIMD iteration**
/// (8 floats = 4 × `[re, im]` pairs).
///
/// # Complex multiply in SIMD
///
/// For each pair `(s, h)` the product is `(s.re·h.re − s.im·h.im, s.re·h.im + s.im·h.re)`.
/// Laid out as 8-wide f32 vectors:
///
/// ```text
/// s_re_dup = [s0r, s0r, s1r, s1r, s2r, s2r, s3r, s3r]   ← re duplicated into pairs
/// s_im_dup = [s0i, s0i, s1i, s1i, s2i, s2i, s3i, s3i]   ← im duplicated into pairs
/// h_ri     = [h0r, h0i, h1r, h1i, h2r, h2i, h3r, h3i]   ← normal interleaved layout
/// h_ni     = [-h0i, h0r, -h1i, h1r, -h2i, h2r, -h3i, h3r] ← negated-im / swapped pairs
///
/// s_re_dup * h_ri + s_im_dup * h_ni
///   = [s0r·h0r + s0i·(−h0i),  s0r·h0i + s0i·h0r,  …]
///   = [s0r·h0r − s0i·h0i,     s0r·h0i + s0i·h0r,  …]   ✓
/// ```
///
/// Compiles to AVX2 FMA on x86-64, NEON on ARM, and SIMD128 on wasm32.
/// The tail (`n % 4` complex) is handled by a scalar fallback.
///
/// # Layout note
///
/// `num_complex::Complex<f32>` is `#[repr(C)]` with two consecutive `f32`
/// fields, so a `[Complex<f32>; N]` has the same memory representation as
/// a `[f32; 2N]` in `[re0, im0, re1, im1, …]` order.  We reinterpret the
/// slices to give the compiler flat `f32` loads (256-bit `vmovups` on AVX)
/// instead of eight separate field accesses per iteration.
#[inline]
fn complex_mac(acc: &mut [Complex<f32>], slot: &[Complex<f32>], ir: &[Complex<f32>]) {
    let n = acc.len();
    let chunks = n / 4;

    // SAFETY: `Complex<f32>` is `#[repr(C)]` with fields `re: f32, im: f32`,
    // so `&[Complex<f32>]` of length N and `&[f32]` of length 2N share the
    // same allocation, alignment (f32 alignment divides Complex<f32>
    // alignment), and lifetime.  We hold no other aliasing reference to
    // these slices for the duration of the returned view.
    let acc_f: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(acc.as_mut_ptr() as *mut f32, n * 2) };
    let slot_f: &[f32] = unsafe { std::slice::from_raw_parts(slot.as_ptr() as *const f32, n * 2) };
    let ir_f: &[f32] = unsafe { std::slice::from_raw_parts(ir.as_ptr() as *const f32, n * 2) };

    for c in 0..chunks {
        let i = c * 8; // 8 floats = 4 complex numbers per iteration

        // Load 4 complex as flat f32 pairs: [re0, im0, re1, im1, re2, im2, re3, im3].
        let s_re = f32x8::from([
            slot_f[i],
            slot_f[i],
            slot_f[i + 2],
            slot_f[i + 2],
            slot_f[i + 4],
            slot_f[i + 4],
            slot_f[i + 6],
            slot_f[i + 6],
        ]);
        let s_im = f32x8::from([
            slot_f[i + 1],
            slot_f[i + 1],
            slot_f[i + 3],
            slot_f[i + 3],
            slot_f[i + 5],
            slot_f[i + 5],
            slot_f[i + 7],
            slot_f[i + 7],
        ]);
        let h_ri = f32x8::from([
            ir_f[i],
            ir_f[i + 1],
            ir_f[i + 2],
            ir_f[i + 3],
            ir_f[i + 4],
            ir_f[i + 5],
            ir_f[i + 6],
            ir_f[i + 7],
        ]);
        // Negate im and swap each pair: [-h.im, h.re, …]
        let h_ni = f32x8::from([
            -ir_f[i + 1],
            ir_f[i],
            -ir_f[i + 3],
            ir_f[i + 2],
            -ir_f[i + 5],
            ir_f[i + 4],
            -ir_f[i + 7],
            ir_f[i + 6],
        ]);
        let a_ri = f32x8::from([
            acc_f[i],
            acc_f[i + 1],
            acc_f[i + 2],
            acc_f[i + 3],
            acc_f[i + 4],
            acc_f[i + 5],
            acc_f[i + 6],
            acc_f[i + 7],
        ]);

        let r = (a_ri + s_re * h_ri + s_im * h_ni).to_array();

        acc_f[i..i + 8].copy_from_slice(&r);
    }

    // Scalar tail for n % 4 remaining bins.
    // In practice spectrum_size = 2 * block_size is always a power-of-two ≥ 4,
    // so this branch is never taken — it exists only for correctness.
    for i in (chunks * 4)..n {
        acc[i] += slot[i] * ir[i];
    }
}

/// Circular buffer of complex frequency-domain spectra for partitioned
/// convolution.
///
/// Stores the `P` most recent input spectra (one pushed per audio block).
/// Use [`push`] to insert the current block's spectrum, then [`accumulate`]
/// to compute `Σ_k slot[k] ⊙ H[k]` across all P IR partitions.
///
/// # Real-time safety
///
/// All methods are allocation-free after construction and safe to call from
/// the audio render thread.
///
/// [`push`]: FreqDomainDelayLine::push
/// [`accumulate`]: FreqDomainDelayLine::accumulate
pub struct FreqDomainDelayLine {
    /// Circular slot buffer, shape `[num_partitions][spectrum_size]`.
    slots: Vec<Vec<Complex<f32>>>,

    /// Index of the **next** slot to write.
    /// The most-recently pushed spectrum is at `(write_pos + P − 1) % P`.
    write_pos: usize,

    /// Number of complex bins per spectrum.  With the real-input FFT
    /// used by [`crate::dsp::FftHandler`] this is `block_size + 1`
    /// (`fft_size / 2 + 1`), i.e. the unique bins of a `2N`-point
    /// real-to-complex transform.
    spectrum_size: usize,
}

impl FreqDomainDelayLine {
    /// Allocate an FDL with `num_partitions` slots, each holding `spectrum_size`
    /// complex bins, initialised to zero.
    ///
    /// **Allocates** — call from a non-real-time thread.
    ///
    /// # Panics
    ///
    /// Panics if `num_partitions == 0` or `spectrum_size == 0`.
    pub fn new(num_partitions: usize, spectrum_size: usize) -> Self {
        assert!(num_partitions > 0, "num_partitions must be > 0");
        assert!(spectrum_size > 0, "spectrum_size must be > 0");
        Self {
            slots: vec![vec![Complex::default(); spectrum_size]; num_partitions],
            write_pos: 0,
            spectrum_size,
        }
    }

    /// Number of delay slots (equals the number of IR partitions).
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.slots.len()
    }

    /// Number of complex bins in each stored spectrum
    /// (`block_size + 1` for the real-input FFT used by the render path).
    #[inline]
    pub fn spectrum_size(&self) -> usize {
        self.spectrum_size
    }

    /// Insert `spectrum` as the newest entry and advance the write pointer.
    ///
    /// After this call `spectrum` is at delay index 0 for the next
    /// [`accumulate`] call.
    ///
    /// `spectrum.len()` must equal [`Self::spectrum_size()`].
    ///
    /// **Real-time safe** — no allocation.
    ///
    /// [`accumulate`]: Self::accumulate
    pub fn push(&mut self, spectrum: &[Complex<f32>]) {
        debug_assert_eq!(
            spectrum.len(),
            self.spectrum_size,
            "push: spectrum length must equal spectrum_size"
        );
        self.slots[self.write_pos].copy_from_slice(spectrum);
        self.write_pos = (self.write_pos + 1) % self.slots.len();
    }

    /// Multiply-accumulate all delay slots against the IR partition spectra,
    /// **adding** the result into `acc`.
    ///
    /// Computes:
    ///
    /// ```text
    /// acc += Σ_{k=0}^{P-1}  slot[k]  ⊙  ir_spectra[k]
    /// ```
    ///
    /// where `slot[0]` is the spectrum from the **most recent** [`push`] and
    /// `slot[k]` is the spectrum from `k` blocks ago.
    ///
    /// `acc` is **not** zeroed before use — zero it beforehand if a fresh
    /// result is needed.
    ///
    /// All slices must have length [`Self::spectrum_size()`], and
    /// `ir_spectra.len()` must equal [`Self::num_partitions()`].
    ///
    /// **Real-time safe** — no allocation.
    ///
    /// [`push`]: Self::push
    pub fn accumulate(&self, ir_spectra: &[Vec<Complex<f32>>], acc: &mut [Complex<f32>]) {
        debug_assert_eq!(
            ir_spectra.len(),
            self.slots.len(),
            "accumulate: ir_spectra.len() must equal num_partitions"
        );
        debug_assert_eq!(
            acc.len(),
            self.spectrum_size,
            "accumulate: acc.len() must equal spectrum_size"
        );

        let p = self.slots.len();

        for (k, ir) in ir_spectra.iter().enumerate() {
            // Slot for delay k: the most-recent push is at write_pos−1 (mod P).
            // Delay k maps to (write_pos + P − 1 − k) % P.
            let slot_idx = (self.write_pos + p - 1 - k) % p;
            let slot = &self.slots[slot_idx];

            complex_mac(acc, slot, ir);
        }
    }

    /// Zero every slot and reset the write pointer.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.fill(Complex::default());
        }
        self.write_pos = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a constant spectrum: every bin equals `value`.
    fn const_spec(spectrum_size: usize, value: Complex<f32>) -> Vec<Complex<f32>> {
        vec![value; spectrum_size]
    }

    /// Build identity IR spectra: each partition is a flat spectrum of ones.
    /// Accumulating against these returns the sum of all stored slots.
    fn identity_ir(num_partitions: usize, spectrum_size: usize) -> Vec<Vec<Complex<f32>>> {
        vec![const_spec(spectrum_size, Complex::new(1.0, 0.0)); num_partitions]
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /// A freshly-created FDL must return all-zero spectra.
    #[test]
    fn new_fdl_is_all_zeros() {
        let fdl = FreqDomainDelayLine::new(3, 8);
        let ir = identity_ir(3, 8);
        let mut acc = vec![Complex::default(); 8];
        fdl.accumulate(&ir, &mut acc);
        for c in &acc {
            assert!(c.norm() < EPSILON, "expected zero, got {c}");
        }
    }

    /// Accessors report the values passed to `new`.
    #[test]
    fn accessors_report_construction_params() {
        let fdl = FreqDomainDelayLine::new(5, 16);
        assert_eq!(fdl.num_partitions(), 5);
        assert_eq!(fdl.spectrum_size(), 16);
    }

    // ── push / accumulate correctness ─────────────────────────────────────────

    /// After one push with identity IR, acc must equal the pushed spectrum.
    #[test]
    fn single_push_identity_ir() {
        let spectrum_size = 4;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);
        let pushed = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        fdl.push(&pushed);

        let ir = identity_ir(1, spectrum_size);
        let mut acc = vec![Complex::default(); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        for (a, &p) in acc.iter().zip(pushed.iter()) {
            assert!((a - p).norm() < EPSILON, "expected {p}, got {a}");
        }
    }

    /// Delay ordering: slot[0] = most recent, slot[1] = one block older.
    ///
    /// Push A then B (B is more recent).  Use selective IR to read each slot:
    ///   IR[0] = [1, 0, 0, 0]   (picks slot[0] = B, but only bin 0)
    ///   IR[1] = [0, 1, 0, 0]   (picks slot[1] = A, but only bin 1)
    ///
    /// Expected: acc[0] = B[0], acc[1] = A[1].
    #[test]
    fn delay_ordering_most_recent_is_slot_zero() {
        let spectrum_size = 4;
        let mut fdl = FreqDomainDelayLine::new(2, spectrum_size);

        let a = vec![
            Complex::new(10.0, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let b = vec![
            Complex::new(30.0, 0.0),
            Complex::new(40.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        fdl.push(&a); // A is older
        fdl.push(&b); // B is most recent

        // IR[0] reads only bin 0 of slot[0] (= B).
        // IR[1] reads only bin 1 of slot[1] (= A).
        let ir = vec![
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        ];

        let mut acc = vec![Complex::default(); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        assert!((acc[0].re - 30.0).abs() < EPSILON, "acc[0]={}", acc[0]);
        assert!((acc[1].re - 20.0).abs() < EPSILON, "acc[1]={}", acc[1]);
        assert!(acc[0].im.abs() < EPSILON);
        assert!(acc[1].im.abs() < EPSILON);
    }

    /// Circular wrap: after P+1 pushes the oldest slot is overwritten.
    ///
    /// P=2. Push A, B, C.  Now slot[0]=B (older), slot[1]=C (most recent).
    /// With identity IR: acc = B + C (A was evicted).
    #[test]
    fn circular_wrap_evicts_oldest_slot() {
        let spectrum_size = 2;
        let mut fdl = FreqDomainDelayLine::new(2, spectrum_size);

        let a = const_spec(spectrum_size, Complex::new(100.0, 0.0));
        let b = const_spec(spectrum_size, Complex::new(2.0, 0.0));
        let c = const_spec(spectrum_size, Complex::new(3.0, 0.0));

        fdl.push(&a); // will be evicted on 4th push
        fdl.push(&b); // older (delay 1)
        fdl.push(&c); // most recent (delay 0)
        // A is now overwritten: only B and C remain.

        let ir = identity_ir(2, spectrum_size);
        let mut acc = vec![Complex::default(); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        // Expected: B + C = 2 + 3 = 5 in each bin.
        for c in &acc {
            assert!((c.re - 5.0).abs() < EPSILON, "expected 5.0, got {}", c.re);
        }
    }

    /// Complex IR entries: accumulate computes correct complex products.
    #[test]
    fn complex_multiply_accumulate() {
        let spectrum_size = 1;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);

        // Push one spectrum: [3 + 4j].
        let spectrum = vec![Complex::new(3.0, 4.0)];
        fdl.push(&spectrum);

        // IR partition: [2 + 1j].
        let ir = vec![vec![Complex::new(2.0, 1.0)]];

        // Expected: (3+4j)*(2+1j) = 6+3j+8j+4j² = (6-4) + (3+8)j = 2 + 11j.
        let mut acc = vec![Complex::default(); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        assert!((acc[0].re - 2.0).abs() < EPSILON, "re={}", acc[0].re);
        assert!((acc[0].im - 11.0).abs() < EPSILON, "im={}", acc[0].im);
    }

    /// accumulate *adds* to acc — pre-existing values are preserved.
    #[test]
    fn accumulate_adds_to_existing_acc() {
        let spectrum_size = 2;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);
        fdl.push(&const_spec(spectrum_size, Complex::new(1.0, 0.0)));

        let ir = identity_ir(1, spectrum_size);
        // Pre-seed acc with 5.
        let mut acc = vec![Complex::new(5.0, 0.0); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        // Each bin: 5 + 1 * 1 = 6.
        for c in &acc {
            assert!((c.re - 6.0).abs() < EPSILON, "expected 6.0, got {}", c.re);
        }
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    /// reset() must zero all slots and restore write_pos to 0 so that the
    /// FDL behaves identically to a freshly-constructed one.
    #[test]
    fn reset_zeros_all_slots() {
        let spectrum_size = 4;
        let mut fdl = FreqDomainDelayLine::new(2, spectrum_size);

        // Push non-zero data.
        fdl.push(&const_spec(spectrum_size, Complex::new(9.0, 9.0)));
        fdl.push(&const_spec(spectrum_size, Complex::new(9.0, 9.0)));
        fdl.reset();

        let ir = identity_ir(2, spectrum_size);
        let mut acc = vec![Complex::default(); spectrum_size];
        fdl.accumulate(&ir, &mut acc);

        for c in &acc {
            assert!(c.norm() < EPSILON, "expected zero after reset, got {c}");
        }
    }

    /// After reset, the FDL produces the same results as a freshly-constructed
    /// one for a given sequence of pushes.
    #[test]
    fn reset_produces_same_output_as_fresh_fdl() {
        let spectrum_size = 4;
        let p = 2;

        let mut fdl = FreqDomainDelayLine::new(p, spectrum_size);
        let mut fresh = FreqDomainDelayLine::new(p, spectrum_size);

        let spec_a = const_spec(spectrum_size, Complex::new(3.0, 1.0));
        let spec_b = const_spec(spectrum_size, Complex::new(7.0, 2.0));

        // Prime the FDL, then reset.
        fdl.push(&const_spec(spectrum_size, Complex::new(99.0, 99.0)));
        fdl.push(&const_spec(spectrum_size, Complex::new(99.0, 99.0)));
        fdl.reset();

        // Feed the same sequence to both.
        fdl.push(&spec_a);
        fresh.push(&spec_a);
        fdl.push(&spec_b);
        fresh.push(&spec_b);

        let ir = identity_ir(p, spectrum_size);
        let mut acc_fdl = vec![Complex::default(); spectrum_size];
        let mut acc_fresh = vec![Complex::default(); spectrum_size];

        fdl.accumulate(&ir, &mut acc_fdl);
        fresh.accumulate(&ir, &mut acc_fresh);

        for (f, r) in acc_fdl.iter().zip(acc_fresh.iter()) {
            assert!(
                (f - r).norm() < EPSILON,
                "reset FDL diverged from fresh: {f} != {r}"
            );
        }
    }
}
