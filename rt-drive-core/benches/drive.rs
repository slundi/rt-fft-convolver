//! Phase-4 micro-benchmarks: each shaper at each oversampling factor.
//!
//! Run with `cargo bench -p rt-drive-core`.  Measures the per-block cost of
//! [`DriveStage::process_block`] — the whole `↑ → shape → ↓ → DC-block → tone`
//! path — for the shipped shaper families across 1× … 16× oversampling.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rt_drive_core::{DriveStage, PedalModel, Shaper, ToneConfig};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 512;

fn pedal(shaper: Shaper, factor: u8) -> PedalModel {
    PedalModel {
        name: "Bench".to_string(),
        input_gain_db: 6.0,
        shaper,
        bias: 0.0,
        tone: ToneConfig::Bypass,
        output_gain_db: 0.0,
        default_oversampling: factor,
    }
}

fn bench_shapers(criterion: &mut Criterion) {
    let shapers: [(&str, Shaper); 4] = [
        ("tanh", Shaper::Tanh),
        ("cubic", Shaper::CubicSoft),
        (
            "hardclip",
            Shaper::HardClip {
                threshold: 0.7,
                asymmetry: 0.2,
            },
        ),
        ("diode", Shaper::Diode { drive: 2.0 }),
    ];

    let mut group = criterion.benchmark_group("drive_stage");
    group.throughput(Throughput::Elements(BLOCK as u64));

    let input: Vec<f32> = (0..BLOCK)
        .map(|n| 0.5 * (std::f32::consts::TAU * 0.01 * n as f32).sin())
        .collect();

    for (name, shaper) in shapers {
        for factor in [1u8, 2, 4, 8, 16] {
            let mut stage = DriveStage::from_model(&pedal(shaper, factor), SAMPLE_RATE, BLOCK);
            let mut output = vec![0.0f32; BLOCK];
            group.bench_function(BenchmarkId::new(name, format!("{factor}x")), |bencher| {
                bencher.iter(|| {
                    stage.process_block(&input, &mut output);
                    std::hint::black_box(&output);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_shapers);
criterion_main!(benches);
