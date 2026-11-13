# Ideas: Extending the Convolver & Adjacent DSP Rewrites

Non-committal brainstorm of directions this library could grow into, and adjacent audio-DSP areas where a Rust rewrite would lower CPU load while keeping feature parity with existing C/C++ tools.

---

## 1. Additional uses for the existing convolution engine

These reuse the current FDL / uniform-partitioned / overlap-save core with mostly IR-management changes on top.

### Guitar-specific
- **Mic IRs** — captures of specific microphones + placements on a cabinet (Celestion, ML Sound Lab style). Same IR pipeline as cab sims; usually just a second convolution stage.
- **Pedal / preamp IRs** — DI-through-pedal captures. Linear portion only; nonlinear stages need a separate model.
- **Acoustic guitar body IRs** — turn a piezo pickup into a mic'd acoustic (ToneWood, ToneDexter style).
- **Nebula-style dynamic convolution** — banks of IRs sampled at multiple input levels (and optionally multiple frequencies) to approximate nonlinear gear. Needs an IR selector + crossfader driven by input level. Reuses `HybridConvolver` and `CrossfadingConvolver` almost verbatim.
- **Reamping IRs** — match the linear transfer function of a full signal chain.

### Broader audio
- **HRTF / binaural rendering** — 3D positioning via head-related impulse responses. Per-source stereo convolution; SOFA file loading; interpolated IR switching for moving sources. Straightforward extension of `stereo.rs`.
- **Room correction** — convolve with an inverted room response (Dirac, Sonarworks equivalents). Needs an inverse-filter/regularisation stage offline; playback is pure convolution.
- **Linear-phase FIR EQ & crossovers** — mastering EQs and multi-band splitters. Long FIRs handled well by partitioned convolution.
- **Match EQ** — transfer a spectral fingerprint from one signal to another. Offline analysis + runtime convolution.
- **Anechoic → space matching** — dubbing a dry recording to match a scene's ambience.
- **De-reverberation** — inverse-filter approaches; shares the runtime engine.

**Verdict:** Nebula-style dynamic convolution and HRTF renderers are the highest-leverage additions — most of the code already exists.

---

## 2. Adjacent DSP: Rust rewrites that would meaningfully lower CPU

These are *not* convolution — they'd be new engines living alongside the current one.

### Matrix

Legend:
- **Complexity**: implementation difficulty and depth of DSP knowledge required.
- **Est. CPU**: rough per-instance CPU on one core at 48 kHz stereo, in a modern DAW. Ranges cover common configurations (e.g. oversampling factor, model size, FFT size). Real numbers depend heavily on implementation quality.
- **ROI**: expected CPU savings from a careful Rust rewrite vs. mainstream C++ implementations.

| # | Effect | Complexity | Est. CPU (per instance) | ROI | Example plugins / hardware |
|---|---|---|---|---|---|
| 1 | Neural amp modeling | High | 3 – 15 % | High | NAM, Proteus (GuitarML), Neural DSP (Nolly, Petrucci, Archetype series), Tonex, Line 6 Helix Native, Kemper Profiler |
| 2 | Oversampled nonlinear stages (drive/saturation) | Medium | 1 – 4 % per stage (4×) | High | Waves GTR3, IK AmpliTube, Klon Centaur / TS808 / ProCo Rat emulations, U-he Satin, Softube Tape |
| 3 | Pitch shift / time stretch | High | 3 – 8 % | High | DigiTech Whammy, Eventide PitchFactor / H9, Boss PS-6, Little Alter Boy, zplane élastique, Waves SoundShifter |
| 4 | Spectral processing (denoise, repair, spectral dynamics) | Very High | 2 – 10 % | High | iZotope RX, Waves NS1 / Clarity Vx, FabFilter Pro-DS, Cedar DNS |
| 5 | FDN / algorithmic reverb | Medium-High | 1 – 4 % | Medium | Valhalla VintageVerb / Room, Strymon BigSky, Eventide Blackhole, Lexicon PCM series, TC Electronic Hall of Fame |
| 6 | Physical / waveguide models | High | 2 – 8 % | Medium | Pianoteq, AAS Chromaphone / String Studio, Line 6 Variax, Roland V-Guitar / V-Drums, Sculpture (Logic) |
| 7 | True-peak limiter (oversampled + look-ahead) | Medium | 1 – 3 % | Medium | FabFilter Pro-L 2, Waves L2 / L3-16, iZotope Ozone Maximizer, Sonnox Oxford Limiter |
| 8 | Formant / vocoder / harmonizer | High | 3 – 10 % | Medium | TC-Helicon VoiceLive, Boss VE-20 / VE-500, DigiTech Vocalist, Antares Harmony Engine, Waves OVox |
| 9 | Adaptive filters (LMS / RLS) | Medium-High | < 1 % | Low | Waves X-FDBK, Sabine FBX, ART Feedback Eliminator, Shure DFR |
| 10 | Loudness metering (EBU R128 / LUFS) | Low | < 0.5 % | Low | Youlean Loudness Meter, iZotope Insight, Waves WLM, TC Clarity M |
| 11 | Multi-band compression / limiting | Medium | 1 – 3 % | Low | FabFilter Pro-MB, Waves C6, iZotope Ozone Dynamics, TDR Nova |

### Notes on the matrix
- **Highest CPU savings potential** (#1–#4): existing implementations frequently violate real-time constraints (audio-thread allocations, locks, cache-unfriendly layouts) and rarely exploit SIMD fully. A disciplined Rust rewrite can be 1.5–3× faster in typical cases.
- **Medium ROI** (#5–#8): mature C++ implementations already exist and are well-optimised; wins come mostly from portability, correctness (denormals, RT-safety), and modernised APIs rather than raw speed.
- **Low ROI** (#9–#11): CPU is already negligible; only worth doing if bundled with something else or needed for feature completeness.

---

## 3. Suggested next steps (if any of this is pursued)

- **Fastest to ship on the current engine:** HRTF renderer (stereo convolver + SOFA loader + IR interpolation).
- **Biggest CPU savings for guitarists:** Neural amp modeling engine. Separate crate, but pairs naturally with this library for the cab stage.
- **Most reusable across products:** Oversampled waveshaper primitives + polyphase resamplers. Foundation for distortion, saturation, tape emulation, tube preamps.

None of these need to happen; this file is a reference for future scoping.
