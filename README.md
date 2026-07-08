# RT FFT Convolver

**rt-fft-convolver** is a professional-grade, real-time digital signal processing (DSP) library for performing fast convolution. It is specifically engineered for guitar cabinet simulation (IR) and reverb effects where ultra-low latency is critical.

## ✨ Features

- **Zero-Latency Hybrid Engine**: Combines Time-Domain Direct Form convolution for the initial impulse response (IR) head with Partitioned FFT convolution for the tail.
- **Optimized FFT**: Powered by `rustfft` for high-performance frequency domain processing.
- **Real-Time Safe**: Zero heap allocations in the processing loop, ensuring glitch-free audio.
- **Anti-Denormal Protection**: Built-in Flush-To-Zero (FTZ) logic to prevent CPU spikes when processing near-silent signals.
- **Multi-Channel Support**: Native support for Mono, Stereo, and True Stereo (4-channel) configurations.
- **Resampling Ready**: Intelligent handling of IR sample rate mismatches.

## Why rt-fft-convolver?

Standard FFT convolution libraries often introduce latency equal to the block size. **rt-fft-convolver** solves this by partitioning the impulse response into smaller segments and using a hybrid approach, allowing guitarists to play through IRs with a perceived latency of under 2ms.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rt-fft-convolver = "0.1"
```

## 🚀 Quick start

The crate exposes composable primitives — assemble the ones you need for
your signal path.  A minimal mono chain using the partitioned FFT engine
(one block of algorithmic latency) plus IR normalisation:

```rust,no_run
use rt_fft_convolver::{
    DenormalGuard, Mixer, UniformPartitionEngine, load_ir,
};

const BLOCK: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Load and resample the IR to the host sample rate ──────────────────
    let ir = load_ir("cab_ir.wav", 48_000)?;
    let samples = &ir.channels[0]; // mono

    // ── FFT-partitioned convolution across the whole IR ───────────────────
    let mut engine = UniformPartitionEngine::new(samples, BLOCK);

    // ── Mixer normalises IR loudness and blends dry / wet ─────────────────
    let mixer = Mixer::new(samples, 1.0); // fully wet

    // Scratch buffers used by the render callback (allocate up front).
    let mut wet = [0.0f32; BLOCK];

    // ── Real-time render callback ─────────────────────────────────────────
    let mut process = move |input: &[f32; BLOCK], output: &mut [f32; BLOCK]| {
        let _guard = DenormalGuard::new(); // FTZ for the whole callback

        wet.fill(0.0);
        engine.process(input, &mut wet);   // convolve x ⊛ ir into wet

        mixer.process(input, &wet, output); // dry × cos + wet × sin·gain
    };

    // Plug `process` into your audio host of choice.
    let (input, mut output) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    process(&input, &mut output);
    Ok(())
}
```

### Zero-latency mode

For applications that need sub-block latency (live guitar rigs), the
[`DirectConvolver`] primitive handles the first `head_len` IR taps in the
time domain with **zero algorithmic latency**.  Assembling it into a true
hybrid engine requires delaying the tail input by `head_len` samples so the
two paths don't double-count — a dedicated `HybridConvolver` type will land
in a future release.  Until then, use [`DirectConvolver`] alone for short
IRs (≤ a few hundred taps) or [`UniformPartitionEngine`] for long IRs.

### Stereo and true-stereo

[`StereoConvolver`] runs an independent engine per channel;
[`TrueStereoConvolver`] runs the full 4-path routing matrix (L→L, L→R, R→L,
R→R) needed for stereo-microphone IR captures.  Both types expose the same
`process(in_l, in_r, out_l, out_r)` signature.

[`DirectConvolver`]: https://docs.rs/rt-fft-convolver/latest/rt_fft_convolver/struct.DirectConvolver.html
[`UniformPartitionEngine`]: https://docs.rs/rt-fft-convolver/latest/rt_fft_convolver/struct.UniformPartitionEngine.html
[`StereoConvolver`]: https://docs.rs/rt-fft-convolver/latest/rt_fft_convolver/struct.StereoConvolver.html
[`TrueStereoConvolver`]: https://docs.rs/rt-fft-convolver/latest/rt_fft_convolver/struct.TrueStereoConvolver.html

## 🤝 Contributing

**rt-fft-convolver** is developed on Codeberg and mirrored to GitHub.

* Primary Repo: [Codeberg main repo](https://codeberg.org/slundi/rt-fft-convolver)
* Mirror: [GitHub repo](https://github.com/slundi/rt-fft-convolver)
