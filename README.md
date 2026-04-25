# RT FFT Convolver

**rt-fft-convolver** is a professional-grade, real-time digital signal processing (DSP) library for performing fast convolution. It is specifically engineered for guitar cabinet simulation (IR) and reverb effects where ultra-low latency is critical.

## Features

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
rust-convolver = "0.1.0"
```

## Quick start

```rust
use rust_convolver::{Convolver, Config};

fn main() {
    // Load your Impulse Response (IR)
    let ir_data = load_wav("cab_ir.wav"); 
    
    // Initialize the engine for 44.1kHz with a 64-sample buffer
    let mut engine = Convolver::new(Config {
        sample_rate: 44100.0,
        partition_size: 64,
        latency_mode: LatencyMode::Zero,
    });

    engine.load_ir(&ir_data);

    // In your audio callback
    audio_host.set_callback(move |buffer| {
        engine.process(buffer);
    });
}
```
