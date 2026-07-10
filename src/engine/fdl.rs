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
//! # Split-complex (structure-of-arrays) storage
//!
//! Every spectrum in this module — the FDL slots, the IR partitions, and the
//! accumulator — is stored as a [`SplitSpectrum`]: the real parts in one
//! contiguous `Vec<f32>` and the imaginary parts in another, rather than the
//! interleaved `[re0, im0, re1, im1, …]` layout of `[Complex<f32>]`.
//!
//! This lets the SIMD complex-multiply kernel issue plain aligned vector loads
//! (`re[i..i+8]`, `im[i..i+8]`) and process **8 complex numbers per iteration**
//! with no in-register shuffles.  The interleaved layout forced either a
//! per-element gather to build the duplicated-real / duplicated-imag vectors,
//! or an in-lane shuffle every iteration; splitting the storage moves that cost
//! out of the O(P) inner loop and into a single deinterleave at [`push`] time
//! (once per block, not once per partition).
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

/// SIMD width: `f32x8` processes 8 real (or 8 imaginary) values per iteration.
/// With split storage that is a full 8 complex numbers per iteration, versus
/// the 4 complex numbers per iteration achievable with interleaved storage.
const LANES: usize = 8;

/// A complex spectrum in **split** (structure-of-arrays) layout: real parts in
/// one contiguous buffer, imaginary parts in another.
///
/// `re[i]` and `im[i]` together form the `i`-th complex bin.  The two vectors
/// always have equal length ([`Self::len`]).
///
/// Storing re/im separately is what makes the complex-MAC kernel a sequence of
/// plain vector loads instead of interleaved shuffles — see the module docs.
#[derive(Clone, Debug, Default)]
pub struct SplitSpectrum {
    /// Real parts, one per bin.
    pub re: Vec<f32>,
    /// Imaginary parts, one per bin (same length as [`Self::re`]).
    pub im: Vec<f32>,
}

impl SplitSpectrum {
    /// Allocate a zero-filled split spectrum of `len` complex bins.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn zeros(len: usize) -> Self {
        Self {
            re: vec![0.0f32; len],
            im: vec![0.0f32; len],
        }
    }

    /// Deinterleave a `&[Complex<f32>]` into a freshly-allocated split spectrum.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn from_complex(src: &[Complex<f32>]) -> Self {
        let mut s = Self::zeros(src.len());
        s.copy_from_complex(src);
        s
    }

    /// Number of complex bins.
    #[inline]
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// `true` if this spectrum holds no bins.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    /// Overwrite this spectrum by deinterleaving `src` into the re/im buffers.
    ///
    /// `src.len()` must equal [`Self::len`].
    ///
    /// **Real-time safe** — no allocation.
    #[inline]
    pub fn copy_from_complex(&mut self, src: &[Complex<f32>]) {
        debug_assert_eq!(
            src.len(),
            self.re.len(),
            "copy_from_complex: length mismatch"
        );
        for (i, c) in src.iter().enumerate() {
            self.re[i] = c.re;
            self.im[i] = c.im;
        }
    }

    /// Interleave this split spectrum back into `dst` as `[Complex<f32>]`.
    ///
    /// `dst.len()` must equal [`Self::len`].
    ///
    /// **Real-time safe** — no allocation.
    #[inline]
    pub fn write_interleaved(&self, dst: &mut [Complex<f32>]) {
        debug_assert_eq!(
            dst.len(),
            self.re.len(),
            "write_interleaved: length mismatch"
        );
        for (i, c) in dst.iter_mut().enumerate() {
            c.re = self.re[i];
            c.im = self.im[i];
        }
    }

    /// Zero both the real and imaginary buffers in place.
    ///
    /// **Real-time safe** — no allocation.
    #[inline]
    pub fn fill_zero(&mut self) {
        self.re.fill(0.0);
        self.im.fill(0.0);
    }
}

/// Load exactly `LANES` contiguous `f32` values into an `f32x8`.
///
/// `s` must have length `LANES`; the slice-to-array conversion is a single
/// aligned load once the optimiser removes the (statically-true) length check.
#[inline(always)]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from(<[f32; LANES]>::try_from(s).expect("load8: slice must be LANES long"))
}

/// Multiply-accumulate `acc += slot * ir` for complex spectra held in split
/// re/im layout.
///
/// # Complex multiply
///
/// For each bin the product is
///
/// ```text
/// (s.re + i·s.im)(h.re + i·h.im)
///   = (s.re·h.re − s.im·h.im) + i·(s.re·h.im + s.im·h.re)
/// ```
///
/// # Why split storage is faster here
///
/// With re and im in separate buffers, `slot_re[i..i+8]`, `slot_im[i..i+8]`,
/// etc. are all contiguous, so each becomes one 256-bit `vmovups`.  The two
/// output lanes then follow directly from four multiplies and two add/subs —
/// no `s.re`/`s.im` duplication and no `[-im, re]` shuffle as the interleaved
/// kernel required.  Eight complex numbers are handled per iteration instead
/// of four.
///
/// Compiles to AVX2 FMA on x86-64, NEON on ARM, and SIMD128 on wasm32.
///
/// # Tail
///
/// `spectrum_size = block_size + 1` is odd for the power-of-two block sizes
/// used by the render path, so the `n % 8` scalar tail always runs for at
/// least the Nyquist bin.
#[inline]
fn complex_mac_split(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    slot_re: &[f32],
    slot_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
) {
    let n = acc_re.len();
    let chunks = n / LANES;

    for c in 0..chunks {
        let i = c * LANES;

        let sr = load8(&slot_re[i..i + LANES]);
        let si = load8(&slot_im[i..i + LANES]);
        let hr = load8(&ir_re[i..i + LANES]);
        let hi = load8(&ir_im[i..i + LANES]);
        let ar = load8(&acc_re[i..i + LANES]);
        let ai = load8(&acc_im[i..i + LANES]);

        // (sr + i·si)(hr + i·hi) = (sr·hr − si·hi) + i·(sr·hi + si·hr)
        let out_re = ar + sr * hr - si * hi;
        let out_im = ai + sr * hi + si * hr;

        acc_re[i..i + LANES].copy_from_slice(&out_re.to_array());
        acc_im[i..i + LANES].copy_from_slice(&out_im.to_array());
    }

    // Scalar tail for the `n % 8` bins the SIMD loop did not cover.
    for i in (chunks * LANES)..n {
        let out_re = acc_re[i] + slot_re[i] * ir_re[i] - slot_im[i] * ir_im[i];
        let out_im = acc_im[i] + slot_re[i] * ir_im[i] + slot_im[i] * ir_re[i];
        acc_re[i] = out_re;
        acc_im[i] = out_im;
    }
}

/// Circular buffer of complex frequency-domain spectra for partitioned
/// convolution.
///
/// Stores the `P` most recent input spectra (one pushed per audio block) in
/// [`SplitSpectrum`] layout.  Use [`push`] to insert the current block's
/// spectrum, then [`accumulate`] to compute `Σ_k slot[k] ⊙ H[k]` across all P
/// IR partitions.
///
/// # Real-time safety
///
/// All methods are allocation-free after construction and safe to call from
/// the audio render thread.
///
/// [`push`]: FreqDomainDelayLine::push
/// [`accumulate`]: FreqDomainDelayLine::accumulate
pub struct FreqDomainDelayLine {
    /// Circular slot buffer, one [`SplitSpectrum`] of `spectrum_size` bins per
    /// slot.
    slots: Vec<SplitSpectrum>,

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
            slots: vec![SplitSpectrum::zeros(spectrum_size); num_partitions],
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
    /// The interleaved `[Complex<f32>]` produced by the forward FFT is
    /// deinterleaved into the target slot's split re/im buffers here — the one
    /// deinterleave per block that keeps [`accumulate`]'s inner loop free of
    /// shuffles.
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
        self.slots[self.write_pos].copy_from_complex(spectrum);
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
    /// `acc` is **not** zeroed before use — call [`SplitSpectrum::fill_zero`]
    /// beforehand if a fresh result is needed.
    ///
    /// All spectra must have length [`Self::spectrum_size()`], and
    /// `ir_spectra.len()` must equal [`Self::num_partitions()`].
    ///
    /// **Real-time safe** — no allocation.
    ///
    /// [`push`]: Self::push
    pub fn accumulate(&self, ir_spectra: &[SplitSpectrum], acc: &mut SplitSpectrum) {
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
        let (acc_re, acc_im) = (&mut acc.re, &mut acc.im);

        for (k, ir) in ir_spectra.iter().enumerate() {
            // Slot for delay k: the most-recent push is at write_pos−1 (mod P).
            // Delay k maps to (write_pos + P − 1 − k) % P.
            let slot_idx = (self.write_pos + p - 1 - k) % p;
            let slot = &self.slots[slot_idx];

            complex_mac_split(acc_re, acc_im, &slot.re, &slot.im, &ir.re, &ir.im);
        }
    }

    /// Zero every slot and reset the write pointer.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.fill_zero();
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

    /// Build a split spectrum where every bin equals `value`.
    fn const_spec(spectrum_size: usize, value: Complex<f32>) -> SplitSpectrum {
        SplitSpectrum::from_complex(&vec![value; spectrum_size])
    }

    /// Build identity IR spectra: each partition is a flat spectrum of ones.
    /// Accumulating against these returns the sum of all stored slots.
    fn identity_ir(num_partitions: usize, spectrum_size: usize) -> Vec<SplitSpectrum> {
        vec![const_spec(spectrum_size, Complex::new(1.0, 0.0)); num_partitions]
    }

    /// Push a `&[Complex]` spectrum from a slice of literals.
    fn push_complex(fdl: &mut FreqDomainDelayLine, values: &[Complex<f32>]) {
        fdl.push(values);
    }

    // ── SplitSpectrum round-trips ───────────────────────────────────────────────

    /// Deinterleave then re-interleave reproduces the original complex slice.
    #[test]
    fn split_spectrum_round_trips_complex() {
        let src = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 4.0),
            Complex::new(5.5, -6.5),
        ];
        let split = SplitSpectrum::from_complex(&src);
        assert_eq!(split.len(), 3);
        assert_eq!(split.re, vec![1.0, -3.0, 5.5]);
        assert_eq!(split.im, vec![2.0, 4.0, -6.5]);

        let mut back = vec![Complex::default(); 3];
        split.write_interleaved(&mut back);
        assert_eq!(back, src);
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /// A freshly-created FDL must return all-zero spectra.
    #[test]
    fn new_fdl_is_all_zeros() {
        let fdl = FreqDomainDelayLine::new(3, 8);
        let ir = identity_ir(3, 8);
        let mut acc = SplitSpectrum::zeros(8);
        fdl.accumulate(&ir, &mut acc);
        for (re, im) in acc.re.iter().zip(acc.im.iter()) {
            assert!(re.abs() < EPSILON && im.abs() < EPSILON, "expected zero");
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
        let pushed = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
        ];
        push_complex(&mut fdl, &pushed);

        let ir = identity_ir(1, spectrum_size);
        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        for (i, &p) in pushed.iter().enumerate() {
            assert!((acc.re[i] - p.re).abs() < EPSILON, "re[{i}]");
            assert!((acc.im[i] - p.im).abs() < EPSILON, "im[{i}]");
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

        let a = [
            Complex::new(10.0, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let b = [
            Complex::new(30.0, 0.0),
            Complex::new(40.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        push_complex(&mut fdl, &a); // A is older
        push_complex(&mut fdl, &b); // B is most recent

        // IR[0] reads only bin 0 of slot[0] (= B).
        // IR[1] reads only bin 1 of slot[1] (= A).
        let ir = vec![
            SplitSpectrum::from_complex(&[
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]),
            SplitSpectrum::from_complex(&[
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]),
        ];

        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        assert!(
            (acc.re[0] - 30.0).abs() < EPSILON,
            "acc.re[0]={}",
            acc.re[0]
        );
        assert!(
            (acc.re[1] - 20.0).abs() < EPSILON,
            "acc.re[1]={}",
            acc.re[1]
        );
        assert!(acc.im[0].abs() < EPSILON);
        assert!(acc.im[1].abs() < EPSILON);
    }

    /// Circular wrap: after P+1 pushes the oldest slot is overwritten.
    ///
    /// P=2. Push A, B, C.  Now slot[0]=B (older), slot[1]=C (most recent).
    /// With identity IR: acc = B + C (A was evicted).
    #[test]
    fn circular_wrap_evicts_oldest_slot() {
        let spectrum_size = 2;
        let mut fdl = FreqDomainDelayLine::new(2, spectrum_size);

        let a = vec![Complex::new(100.0, 0.0); spectrum_size];
        let b = vec![Complex::new(2.0, 0.0); spectrum_size];
        let c = vec![Complex::new(3.0, 0.0); spectrum_size];

        push_complex(&mut fdl, &a); // will be evicted on 4th push
        push_complex(&mut fdl, &b); // older (delay 1)
        push_complex(&mut fdl, &c); // most recent (delay 0)
        // A is now overwritten: only B and C remain.

        let ir = identity_ir(2, spectrum_size);
        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        // Expected: B + C = 2 + 3 = 5 in each bin.
        for &re in &acc.re {
            assert!((re - 5.0).abs() < EPSILON, "expected 5.0, got {re}");
        }
    }

    /// Complex IR entries: accumulate computes correct complex products.
    #[test]
    fn complex_multiply_accumulate() {
        let spectrum_size = 1;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);

        // Push one spectrum: [3 + 4j].
        push_complex(&mut fdl, &[Complex::new(3.0, 4.0)]);

        // IR partition: [2 + 1j].
        let ir = vec![SplitSpectrum::from_complex(&[Complex::new(2.0, 1.0)])];

        // Expected: (3+4j)*(2+1j) = 6+3j+8j+4j² = (6-4) + (3+8)j = 2 + 11j.
        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        assert!((acc.re[0] - 2.0).abs() < EPSILON, "re={}", acc.re[0]);
        assert!((acc.im[0] - 11.0).abs() < EPSILON, "im={}", acc.im[0]);
    }

    /// accumulate *adds* to acc — pre-existing values are preserved.
    #[test]
    fn accumulate_adds_to_existing_acc() {
        let spectrum_size = 2;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);
        push_complex(&mut fdl, &vec![Complex::new(1.0, 0.0); spectrum_size]);

        let ir = identity_ir(1, spectrum_size);
        // Pre-seed acc with 5.
        let mut acc = const_spec(spectrum_size, Complex::new(5.0, 0.0));
        fdl.accumulate(&ir, &mut acc);

        // Each bin: 5 + 1 * 1 = 6.
        for &re in &acc.re {
            assert!((re - 6.0).abs() < EPSILON, "expected 6.0, got {re}");
        }
    }

    /// The SIMD kernel must agree with a scalar reference across a length that
    /// exercises both the 8-wide body and the `n % 8` tail (13 = 8 + 5).
    #[test]
    fn simd_kernel_matches_scalar_reference_with_tail() {
        let spectrum_size = 13;
        let mut fdl = FreqDomainDelayLine::new(1, spectrum_size);

        // Deterministic pseudo-values for slot and IR.
        let slot: Vec<Complex<f32>> = (0..spectrum_size)
            .map(|i| Complex::new(i as f32 * 0.5 - 2.0, 1.0 - i as f32 * 0.25))
            .collect();
        let ir_c: Vec<Complex<f32>> = (0..spectrum_size)
            .map(|i| Complex::new(0.3 * i as f32, -0.7 + i as f32 * 0.1))
            .collect();

        push_complex(&mut fdl, &slot);
        let ir = vec![SplitSpectrum::from_complex(&ir_c)];

        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        for i in 0..spectrum_size {
            let expected = slot[i] * ir_c[i];
            assert!((acc.re[i] - expected.re).abs() < EPSILON, "re[{i}]");
            assert!((acc.im[i] - expected.im).abs() < EPSILON, "im[{i}]");
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
        push_complex(&mut fdl, &vec![Complex::new(9.0, 9.0); spectrum_size]);
        push_complex(&mut fdl, &vec![Complex::new(9.0, 9.0); spectrum_size]);
        fdl.reset();

        let ir = identity_ir(2, spectrum_size);
        let mut acc = SplitSpectrum::zeros(spectrum_size);
        fdl.accumulate(&ir, &mut acc);

        for (re, im) in acc.re.iter().zip(acc.im.iter()) {
            assert!(
                re.abs() < EPSILON && im.abs() < EPSILON,
                "expected zero after reset"
            );
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

        let spec_a = vec![Complex::new(3.0, 1.0); spectrum_size];
        let spec_b = vec![Complex::new(7.0, 2.0); spectrum_size];

        // Prime the FDL, then reset.
        push_complex(&mut fdl, &vec![Complex::new(99.0, 99.0); spectrum_size]);
        push_complex(&mut fdl, &vec![Complex::new(99.0, 99.0); spectrum_size]);
        fdl.reset();

        // Feed the same sequence to both.
        push_complex(&mut fdl, &spec_a);
        push_complex(&mut fresh, &spec_a);
        push_complex(&mut fdl, &spec_b);
        push_complex(&mut fresh, &spec_b);

        let ir = identity_ir(p, spectrum_size);
        let mut acc_fdl = SplitSpectrum::zeros(spectrum_size);
        let mut acc_fresh = SplitSpectrum::zeros(spectrum_size);

        fdl.accumulate(&ir, &mut acc_fdl);
        fresh.accumulate(&ir, &mut acc_fresh);

        for i in 0..spectrum_size {
            assert!(
                (acc_fdl.re[i] - acc_fresh.re[i]).abs() < EPSILON
                    && (acc_fdl.im[i] - acc_fresh.im[i]).abs() < EPSILON,
                "reset FDL diverged from fresh at bin {i}"
            );
        }
    }
}
