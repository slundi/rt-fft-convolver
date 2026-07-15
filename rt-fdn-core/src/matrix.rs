//! The FDN feedback mixing matrix — a fast Walsh–Hadamard transform.
//!
//! # Why a matrix at all?
//!
//! An FDN is `N` delay lines whose outputs are cross-mixed and fed back.  The
//! mixing matrix `A` is what turns `N` independent comb filters into a single
//! dense, diffuse reverb: it scatters each delay's energy across *all* the
//! others every pass.  For the tail to neither grow nor collapse, `A` must be
//! **lossless** — orthonormal (`AᵀA = I`), so it only rotates the energy vector,
//! never changes its length.  The per-line decay is then applied separately, by
//! the feedback gains, which is what lets us dial T60 without touching
//! diffusion.
//!
//! # Why Hadamard (Phase 0 decision)
//!
//! For `N` a power of two the Hadamard matrix is orthonormal and its transform
//! is the **Fast Walsh–Hadamard Transform**: `O(N log N)` using only additions
//! and subtractions — *no multiplies* in the butterfly, one scalar `1/√N`
//! normalisation at the end.  At `N = 8` that is 24 add/subs versus 64
//! multiply-adds for a dense matrix, and it is maximally diffusive (every input
//! reaches every output with equal weight).  See
//! `docs/Roadmap-05-fdn-reverb.md`.
//!
//! # Real-time safety
//!
//! Operates in place on a stack array — no allocation, no locking.

/// Apply the orthonormal Hadamard mix to `vector` in place.
///
/// `N` **must be a power of two** (checked with a `debug_assert`; the FDN only
/// ever instantiates power-of-two sizes).  The result is scaled by `1/√N` so
/// the transform is energy-preserving (`‖Hx‖ = ‖x‖`).
///
/// **Real-time safe.**
#[inline]
pub fn hadamard_in_place<const N: usize>(vector: &mut [f32; N]) {
    debug_assert!(
        N.is_power_of_two() && N > 0,
        "Hadamard size must be a power of two, got {N}"
    );

    // In-place decimation-in-time butterfly: at each stage `half` we combine
    // pairs (j, j+half) as sum / difference.  This is the FWHT in natural order.
    let mut half = 1;
    while half < N {
        let mut block = 0;
        while block < N {
            for offset in block..block + half {
                let a = vector[offset];
                let b = vector[offset + half];
                vector[offset] = a + b;
                vector[offset + half] = a - b;
            }
            block += half << 1;
        }
        half <<= 1;
    }

    // Normalise so the transform is orthonormal (unitary), preserving energy.
    let norm = 1.0 / (N as f32).sqrt();
    for element in vector.iter_mut() {
        *element *= norm;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn energy<const N: usize>(vector: &[f32; N]) -> f32 {
        vector.iter().map(|&x| x * x).sum()
    }

    /// The transform must preserve energy (orthonormality).
    #[test]
    fn preserves_energy_n8() {
        let mut vector = [0.3, -0.7, 1.0, -1.0, 0.25, 0.5, -0.5, 0.1];
        let before = energy(&vector);
        hadamard_in_place(&mut vector);
        let after = energy(&vector);
        assert!((before - after).abs() < 1e-5, "energy {before} → {after}");
    }

    /// Applying the (self-inverse, orthonormal) transform twice returns the
    /// original vector: `H` is its own inverse because `Hᵀ = H` and `HᵀH = I`.
    #[test]
    fn is_self_inverse_n8() {
        let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut vector = original;
        hadamard_in_place(&mut vector);
        hadamard_in_place(&mut vector);
        for (got, want) in vector.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-4, "round-trip {got} != {want}");
        }
    }

    /// A DC input maps all its energy to bin 0 and nowhere else.
    #[test]
    fn dc_maps_to_first_bin() {
        let mut vector = [1.0f32; 8];
        hadamard_in_place(&mut vector);
        // Σ = 8, scaled by 1/√8 → √8 in bin 0, zeros elsewhere.
        assert!((vector[0] - (8.0f32).sqrt()).abs() < 1e-5);
        for &element in &vector[1..] {
            assert!(element.abs() < 1e-5, "off-DC leakage {element}");
        }
    }

    /// Also works at other power-of-two sizes (matrix code is size-generic).
    #[test]
    fn preserves_energy_n4() {
        let mut vector = [1.0, -2.0, 3.0, -4.0];
        let before = energy(&vector);
        hadamard_in_place(&mut vector);
        assert!((before - energy(&vector)).abs() < 1e-5);
    }

    /// Distinct impulses map to distinct rows (the mix is not degenerate).
    #[test]
    fn distinct_impulses_spread_fully() {
        let mut e0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        hadamard_in_place(&mut e0);
        // Every output has equal magnitude 1/√8 — full diffusion.
        let expected = 1.0 / (8.0f32).sqrt();
        for &element in &e0 {
            assert!((element.abs() - expected).abs() < 1e-5, "{element}");
        }
    }
}
