# rt-ir-reverb — User Manual

A **convolution reverb** with true-stereo support.  Load a room / hall / plate
impulse response and it convolves your stereo signal, with eight
instantly-switchable slots and independent dry / wet faders.

## Formats & installation

Ships as **CLAP** and **VST3** (Linux, macOS, Windows) — same install locations
as [rt-ir-cab](manual-cab.md#formats--installation).

## I/O

Stereo → stereo.  The engine picks the convolver from the IR's channel count:

- **4-channel** IR → **true-stereo** (`L→L`, `L→R`, `R→L`, `R→R` cross-feed).
- **1–3-channel** IR → plain stereo (mono IRs are duplicated to both channels).

The plugin reports its processing latency to the host for automatic
compensation.

## Slots

Eight pre-loaded slots, identical behaviour to the cab: drag-to-load, click /
automation / MIDI PC / CC to activate, per-slot enable, empty = dry, and a
click-free **crossfade** on switch (up to 500 ms — reverb tails need the room).

## Controls

| Control        | Range        | Default | Notes                                    |
|----------------|--------------|---------|------------------------------------------|
| Current Slot   | 1–8          | 1       | Automatable; also MIDI PC / CC.          |
| Crossfade Time | 5–500 ms     | 100 ms  | Slot-to-slot transition length.          |
| Pre-Delay      | 0–500 ms     | 0 ms    | Delay before the reverb onset.           |
| Dry            | −∞…+6 dB     | 0 dB    | Independent dry fader.                    |
| Wet            | −∞…+6 dB     | 0 dB    | Independent wet fader.                    |
| Stereo Width   | 0–200 %      | 100 %   | Wet-only mid/side width.                  |
| Low Cut        | 20–500 Hz    | 20 Hz   | Wet-only high-pass.                       |
| High Cut       | 2–20 kHz     | 20 kHz  | Wet-only low-pass.                        |
| IR Reverse     | on / off     | off     | Reverses the IR (reverse-reverb effect).  |
| IR Length      | 1–100 %      | 100 %   | Truncates the tail to trim CPU.           |
| Bypass         | on / off     | off     | Passes the input untouched.               |

Wet flow: `in → Pre-Delay → convolve → Low/High Cut → Stereo Width`, then
`out = Dry·in + Wet·(wet)`.

## Presets

Space presets shape pre-delay, width, EQ, and tail length (**not** the IR
itself — pair each with its room IR):
**Small Room**, **Large Room**, **Hall**, **Plate**, **Spring**.

## MIDI

Same as the cab: Program Change / assignable CC select a slot; the header LED
blinks on incoming events.
