# rt-ir-cab — User Manual

A zero-latency guitar / bass **cabinet simulator**.  Load an impulse response
(IR) of a speaker cabinet and it convolves your dry signal in real time, with
eight instantly-switchable slots for live use.

## Formats & installation

Ships as **CLAP** and **VST3** (Linux, macOS, Windows).  Install with the
platform package (see [packaging](../packaging/README.md)) or copy the bundle:

| Format | Linux | macOS | Windows |
|--------|-------|-------|---------|
| CLAP   | `~/.clap` or `/usr/lib/clap` | `~/Library/Audio/Plug-Ins/CLAP` | `Common Files\CLAP` |
| VST3   | `~/.vst3` or `/usr/lib/vst3` | `~/Library/Audio/Plug-Ins/VST3` | `Common Files\VST3` |

## I/O and latency

Mono → stereo (the guitar case) or stereo → stereo.  A `HybridConvolver` runs
per output channel.  The plugin reports its processing latency to the host, so
your DAW compensates automatically.

## Slots

Eight IR slots, all pre-loaded so switching never drops out on stage.

- **Load**: drag a WAV onto a pad (the file path is saved with the project).
- **Activate**: click a pad, automate `Current Slot`, or send a MIDI Program
  Change / assigned CC.
- **Slot Enabled N**: disable a slot to play dry without unloading it.
- An **empty slot passes the dry signal** — it never goes silent.

Switching slots equal-power **crossfades** over `Crossfade Time`, so patch
changes are click-free.

## Controls

| Control        | Range        | Default | Notes                                  |
|----------------|--------------|---------|----------------------------------------|
| Current Slot   | 1–8          | 1       | Automatable; also MIDI PC / CC.        |
| Crossfade Time | 5–200 ms     | 20 ms   | Slot-to-slot transition length.        |
| Input Gain     | −24…+24 dB   | 0 dB    | Before the cab.                        |
| Low Cut        | 20–500 Hz    | 20 Hz   | 12 dB/oct high-pass, pre-convolution.  |
| High Cut       | 2–20 kHz     | 20 kHz  | 12 dB/oct low-pass, post-convolution.  |
| Mix            | 0–100 %      | 100 %   | 0 % = fully dry, 100 % = fully cab.    |
| Output Gain    | −24…+24 dB   | 0 dB    | After the mix.                         |
| IR Normalize   | on / off     | off     | Level-matches IRs (L2-norm).           |
| Bypass         | on / off     | off     | Passes the input untouched.            |

Signal flow: `in → Input Gain → Low Cut → cabinet → High Cut → Mix → Output Gain`.

## Presets

Tone presets shape **only the EQ** (never the IR, gains, or mix):
**Flat**, **Bright**, **Dark**, **Scooped**.

## MIDI

- **Program Change** *n* selects slot *n*.
- An assignable **CC** selects a slot (value ÷ 16 → slot).
- The header **MIDI LED** blinks on incoming events.
