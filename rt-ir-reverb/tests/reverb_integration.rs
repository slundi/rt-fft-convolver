//! End-to-end integration tests driving `rt-ir-reverb` through its public API.

use rt_ir_reverb::core::{PreparedIr, SlotEvent};
use rt_ir_reverb::{ReverbController, ReverbParams, StereoBlockAdapter};

const BLOCK: usize = 32;
const SR: u32 = 48_000;

fn stereo_impulse(g: f32) -> PreparedIr {
    let mut l = vec![0.0f32; BLOCK];
    l[0] = g;
    let mut r = vec![0.0f32; BLOCK];
    r[0] = g;
    PreparedIr::from_channels(vec![l, r], SR)
}

fn flat_params() -> ReverbParams {
    ReverbParams {
        crossfade_ms: 0.0,
        low_cut_hz: 20.0,
        high_cut_hz: 20_000.0,
        ..ReverbParams::default()
    }
}

/// Render one host buffer through the adapter (in place, stereo).
fn render(adapter: &mut StereoBlockAdapter, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut wl = l.to_vec();
    let mut wr = r.to_vec();
    {
        let mut chans: [&mut [f32]; 2] = [&mut wl, &mut wr];
        adapter.process_buffer(&mut chans);
    }
    (wl, wr)
}

#[test]
fn loaded_reverb_produces_finite_audio_after_latency() {
    let params = flat_params();
    let (mut controller, processor) = ReverbController::new(BLOCK, SR, &params);
    controller.load_slot(0, stereo_impulse(1.0)).unwrap();
    controller.activate(0);

    let mut adapter = StereoBlockAdapter::new(processor);
    assert_eq!(adapter.latency_samples(), BLOCK);

    let input = vec![0.2f32; BLOCK];
    let (l0, _) = render(&mut adapter, &input, &input);
    assert!(l0.iter().all(|&s| s == 0.0), "first block is latency");

    let (l1, r1) = render(&mut adapter, &input, &input);
    assert!(l1.iter().all(|&s| s.is_finite()));
    assert!(r1.iter().all(|&s| s.is_finite()));
    assert!(
        l1.iter().any(|&s| s.abs() > 1e-4),
        "expected audible output"
    );
}

#[test]
fn empty_bank_is_dry_passthrough_delayed() {
    let params = flat_params();
    let (_controller, processor) = ReverbController::new(BLOCK, SR, &params);
    let mut adapter = StereoBlockAdapter::new(processor);

    let input: Vec<f32> = (0..BLOCK).map(|i| i as f32 * 0.01).collect();
    render(&mut adapter, &input, &input); // latency block
    let (l, r) = render(&mut adapter, &input, &input);
    for (i, &want) in input.iter().enumerate() {
        assert!((l[i] - want).abs() < 1e-6, "L sample {i}");
        assert!((r[i] - want).abs() < 1e-6, "R sample {i}");
    }
}

#[test]
fn midi_program_change_selects_slot() {
    let params = flat_params();
    let (mut controller, processor) = ReverbController::new(BLOCK, SR, &params);
    controller.load_slot(6, stereo_impulse(1.0)).unwrap();
    assert_eq!(controller.handle_midi(SlotEvent::ProgramChange(6)), Some(6));

    let mut adapter = StereoBlockAdapter::new(processor);
    let input = vec![0.1f32; BLOCK];
    render(&mut adapter, &input, &input);
    render(&mut adapter, &input, &input);
    assert_eq!(adapter.processor_mut().active_slot(), 6);
}
