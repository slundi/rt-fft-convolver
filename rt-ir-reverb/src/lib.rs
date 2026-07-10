//! `rt-ir-reverb` — convolution reverb plugin with true-stereo support.
//!
//! **Phase 2 placeholder.** The Phase-0 shared foundation lives in
//! [`rt_ir_plugin_core`]; this crate will add the `nih-plug` integration,
//! [`StereoConvolver`](rt_fft_convolver::StereoConvolver) /
//! [`TrueStereoConvolver`](rt_fft_convolver::TrueStereoConvolver)-based I/O,
//! pre-delay, dry/wet faders, and presets described in `docs/Roadmap-lv2.md`.

// Re-exported so the shared foundation is reachable (and the dependency is
// exercised) until the real plugin lands.
pub use rt_ir_plugin_core as core;
