//! End-to-end integration tests driving `rt-ir-cab` through its public API:
//! build the controller/processor pair, load an IR, and render audio through
//! the fixed-block adapter exactly as the plugin wrapper does.

use rt_ir_cab::core::{PreparedIr, SlotEvent};
use rt_ir_cab::{CabController, CabParams, StereoBlockAdapter};

const BLOCK: usize = 32;
const SR: u32 = 48_000;

/// Scaled unit-impulse IR (convolution = multiply by `gain`).
fn impulse_ir(gain: f32) -> PreparedIr {
    let mut ir = vec![0.0f32; BLOCK];
    ir[0] = gain;
    PreparedIr::from_channels(vec![ir], SR)
}

fn flat_params() -> CabParams {
    CabParams {
        crossfade_ms: 0.0,
        low_cut_hz: 20.0,
        high_cut_hz: 20_000.0,
        ..CabParams::default()
    }
}

/// Drive one host-sized buffer through the adapter in place (mono in).
fn render(adapter: &mut StereoBlockAdapter, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut l = input.to_vec();
    let mut r = vec![0.0f32; input.len()];
    {
        let mut chans: [&mut [f32]; 2] = [&mut l, &mut r];
        adapter.process_buffer(&mut chans);
    }
    (l, r)
}

#[test]
fn loaded_cab_produces_finite_audio_after_latency() {
    let params = flat_params();
    let (mut controller, processor) = CabController::new(BLOCK, SR, 2, &params);
    controller.load_slot(0, impulse_ir(1.0)).unwrap();
    controller.activate(0);

    let mut adapter = StereoBlockAdapter::new(processor, false);
    assert_eq!(adapter.latency_samples(), BLOCK);

    // A steady input; first block is latency (silence), later blocks carry audio.
    let input = vec![0.25f32; BLOCK];
    let (l0, _) = render(&mut adapter, &input);
    assert!(
        l0.iter().all(|&s| s == 0.0),
        "first block should be latency"
    );

    let (l1, r1) = render(&mut adapter, &input);
    assert!(l1.iter().all(|&s| s.is_finite()));
    assert!(r1.iter().all(|&s| s.is_finite()));
    // Cab is active → the wet signal is non-trivial once past the latency.
    assert!(
        l1.iter().any(|&s| s.abs() > 1e-4),
        "expected audible output"
    );
}

#[test]
fn midi_program_change_selects_a_loaded_slot() {
    let params = flat_params();
    let (mut controller, processor) = CabController::new(BLOCK, SR, 2, &params);
    controller.load_slot(5, impulse_ir(2.0)).unwrap();

    // Program Change 5 routes to slot 5.
    assert_eq!(controller.handle_midi(SlotEvent::ProgramChange(5)), Some(5));

    let mut adapter = StereoBlockAdapter::new(processor, false);
    let input = vec![0.1f32; BLOCK];
    // Prime past the latency block, then confirm the processor tracks slot 5.
    render(&mut adapter, &input);
    render(&mut adapter, &input);
    assert_eq!(adapter.processor_mut().active_slot(), 5);
}

#[test]
fn empty_bank_is_dry_passthrough_delayed() {
    // No IR loaded anywhere → the plugin must pass audio (never silence),
    // delayed by the adapter latency.
    let params = flat_params();
    let (_controller, processor) = CabController::new(BLOCK, SR, 2, &params);
    let mut adapter = StereoBlockAdapter::new(processor, false);

    let input: Vec<f32> = (0..BLOCK).map(|i| (i as f32) * 0.01).collect();
    render(&mut adapter, &input); // latency block
    let (l, r) = render(&mut adapter, &input);
    // Dry passthrough at unity → the second block equals the input.
    for (i, (&got, &want)) in l.iter().zip(input.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "sample {i}: {got} != {want}");
    }
    assert_eq!(l, r);
}
