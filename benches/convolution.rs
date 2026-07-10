//! Criterion benchmark harness for the partitioned-convolution hot path.
//!
//! The frequency-domain multiply-accumulate (complex-MAC) over all IR
//! partitions dominates the per-block cost of [`UniformPartitionEngine`], so
//! this harness measures it two ways:
//!
//! * `partition_process` — the full end-to-end block: overlap-save slide,
//!   forward FFT, FDL push, complex-MAC, inverse FFT.  This is what the audio
//!   render thread actually calls.
//! * `fdl_accumulate` — the complex-MAC in isolation (the step the
//!   split-real/imag storage optimisation targets), so a change to the kernel
//!   is visible without the FFT cost dominating the measurement.
//!
//! Run with `cargo bench`.  The IR / block sizes below mirror a typical guitar
//! cabinet IR (a few thousand taps) processed at common audio block sizes.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustfft::num_complex::Complex;

use rt_fft_convolver::{FreqDomainDelayLine, SplitSpectrum, UniformPartitionEngine};

/// Deterministic pseudo-random-ish IR tail of `len` taps (a decaying,
/// oscillating signal — no `rand` dependency, and stable across runs).
fn make_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32;
            (t * 0.017).sin() * (-t / (len as f32 * 0.4)).exp()
        })
        .collect()
}

/// A block of `n` deterministic input samples.
fn make_block(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect()
}

/// End-to-end `process()` cost for a range of block sizes at a fixed IR length.
fn bench_partition_process(c: &mut Criterion) {
    const IR_LEN: usize = 8_192; // ~170 ms tail at 48 kHz

    let mut group = c.benchmark_group("partition_process");
    for &block_size in &[64usize, 128, 256, 512] {
        let ir = make_ir(IR_LEN);
        let input = make_block(block_size);
        let mut engine = UniformPartitionEngine::new(&ir, block_size);
        let mut output = vec![0.0f32; block_size];

        group.throughput(Throughput::Elements(block_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(block_size),
            &block_size,
            |b, _| {
                b.iter(|| {
                    output.fill(0.0);
                    engine.process(black_box(&input), black_box(&mut output));
                });
            },
        );
    }
    group.finish();
}

/// Isolated complex-MAC (`FreqDomainDelayLine::accumulate`) — the step the
/// split re/im storage optimises — across a range of partition counts.
fn bench_fdl_accumulate(c: &mut Criterion) {
    const BLOCK_SIZE: usize = 256;
    let spectrum_size = BLOCK_SIZE + 1;

    let mut group = c.benchmark_group("fdl_accumulate");
    for &num_partitions in &[8usize, 32, 128] {
        // Prime the FDL with `num_partitions` deterministic spectra.
        let mut fdl = FreqDomainDelayLine::new(num_partitions, spectrum_size);
        for k in 0..num_partitions {
            let spec: Vec<Complex<f32>> = (0..spectrum_size)
                .map(|i| Complex::new((i + k) as f32 * 0.1, (i * 2 + k) as f32 * 0.05))
                .collect();
            fdl.push(&spec);
        }

        // IR partitions in split layout.
        let ir: Vec<SplitSpectrum> = (0..num_partitions)
            .map(|k| {
                let spec: Vec<Complex<f32>> = (0..spectrum_size)
                    .map(|i| Complex::new(0.2 - i as f32 * 0.001, 0.1 + k as f32 * 0.01))
                    .collect();
                SplitSpectrum::from_complex(&spec)
            })
            .collect();

        let mut acc = SplitSpectrum::zeros(spectrum_size);

        group.throughput(Throughput::Elements(
            (num_partitions * spectrum_size) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_partitions),
            &num_partitions,
            |b, _| {
                b.iter(|| {
                    acc.fill_zero();
                    fdl.accumulate(black_box(&ir), black_box(&mut acc));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_partition_process, bench_fdl_accumulate);
criterion_main!(benches);
