# UI Design: `rt-ir-cab` & `rt-ir-reverb`

This document specifies the user interface for both LV2 plugins.  It
covers the shared design language, per-plugin layouts, component
states, interactions, and accessibility.  Companion to
[`Roadmap-lv2.md`](Roadmap-lv2.md).

## Guiding principles

1. **Single-pane, no tabs**.  Every control is visible at once.  If a
   control needs a panel, it is not important enough to ship in 1.0.
2. **Stage-safe defaults**.  Big hit-targets, high-contrast active
   states, no destructive-by-default actions.  A guitarist glancing
   at a tablet on a dark stage must be able to see which slot is
   active from two metres away.
3. **Host-native feel**.  Bypass, undo, MIDI learn, and preset
   management delegate to the host wherever possible.  We don't
   reinvent DAW plumbing.
4. **Two plugins, one visual language**.  Shared components, tokens,
   and layout grid.  Only the accent color and a handful of
   plugin-specific controls differ.

## Design tokens

Shared across both plugins.  Values are proposed defaults — adjust
during implementation with `nih-plug-egui`'s theming.

### Color palette

| Token             | Value            | Usage                             |
|-------------------|------------------|-----------------------------------|
| `bg.base`         | `#111418`        | Window background                 |
| `bg.panel`        | `#1a1e24`        | Slot rack, waveform frame         |
| `bg.raised`       | `#232830`        | Knob wells, fader tracks          |
| `fg.primary`      | `#e6e8ec`        | Labels, values                    |
| `fg.secondary`    | `#8a92a0`        | Units, secondary text             |
| `fg.disabled`     | `#4a5060`        | Empty-slot text                   |
| `border.default`  | `#2c323c`        | Component borders                 |
| `border.strong`   | `#3f4655`        | Slot borders                      |
| `accent.cab`      | `#e0a144`        | `rt-ir-cab` accent (amber)        |
| `accent.reverb`   | `#3aa3b5`        | `rt-ir-reverb` accent (teal)      |
| `signal.midi.ok`  | `#4ec36a`        | MIDI activity LED (recognised)    |
| `signal.midi.err` | `#d94a4a`        | MIDI activity LED (unrecognised)  |
| `signal.clip`     | `#d94a4a`        | Clip warning on gain/output       |

### Typography

- **Body / labels**: Inter 12 px, `fg.secondary` for labels,
  `fg.primary` for values.
- **Slot names**: Inter 11 px, uppercase, letter-spaced.
- **Header / plugin name**: Inter 14 px semi-bold.
- **Value readouts under knobs**: JetBrains Mono 11 px — monospaced
  digits stop values jittering horizontally as they change.

### Spacing grid

4 px base unit.  All gaps, paddings, and component sizes snap to
multiples of 4.  Component-internal padding: 8 px.  Section gaps:
16 px.  Window edge inset: 12 px.

### Component dimensions

| Component        | Size (px)            |
|------------------|----------------------|
| Slot pad         | 64 × 56              |
| Slot rack gap    | 8                    |
| Knob             | 48 × 48 + 16 label   |
| Fader            | 24 × 100             |
| Header bar       | full-width × 32      |
| Waveform frame   | full-width × 96      |
| Bypass button    | 96 × 28              |
| PANIC button     | 72 × 28              |

## Shared components

### Slot pad

The 8-slot rack is the identity component of both plugins.  Same
behaviour, same states, in both.

| State              | Border           | Background        | Text          |
|--------------------|------------------|-------------------|---------------|
| Empty              | dashed, `border.default` | `bg.panel` | `fg.disabled` "·" |
| Filled + inactive  | solid, `border.strong`   | `bg.panel` | `fg.primary` name |
| Filled + active    | 2 px `accent.*`  | `accent.*` @ 12 % | `fg.primary` name + accent underline |
| Loading            | solid, `border.strong`   | `bg.panel` | animated dots |
| Hover (any state)  | +1 px inner glow | +4 % lightness    | unchanged     |

**Interactions**:

- **Left-click**: activate this slot (empty slots are no-op).
- **Double-click**: open file picker.
- **Right-click**: context menu — Load, Rename, Clear, Reveal in
  file browser, MIDI Learn.
- **Drag WAV/FLAC/AIFF onto the pad**: load into that slot.  Any
  drag while another slot is being loaded is queued, not dropped.
- **Keyboard**: arrow keys move focus across slots; Enter activates;
  Delete clears.

### Knob

Circular indicator with a value readout below.  Values shown in
their native unit (dB, Hz, %, ms).

- **Drag vertical** to change value.  Hold Shift for fine control
  (× 0.1 sensitivity).  Hold Ctrl to snap to integer / nearest whole
  dB / whole Hz.
- **Double-click** resets to default.
- **Right-click** menu: Enter value, Reset, MIDI Learn.
- **Scroll wheel** also adjusts (same sensitivity as drag).

### Fader (`rt-ir-reverb` only)

Vertical.  Used for Dry and Wet independently.  Includes a fine
scale reading in dB to the right.

- **Drag** to change.  Shift for fine.
- **Double-click** resets to unity / bypass level.
- **Right-click** → same menu as knob.

### Waveform display

Frame around a full-height rendering of the currently active IR.

- **Mono IRs**: single trace, `fg.primary`.
- **Stereo IRs**: two traces stacked, both `fg.primary`.
- **True-stereo IRs (`rt-ir-reverb`)**: four traces in 2 × 2 (L→L,
  L→R / R→L, R→R) with per-trace labels.
- **Amplitude**: normalised to the IR peak.
- **Time axis**: linear.  Show duration on the info line under the
  waveform.
- **Hover over a *different* slot** in the rack: waveform dims to
  60 % opacity and shows a preview of that slot's IR.  Releasing
  the mouse restores the active slot's waveform without switching.

### Header bar

Full-width strip at the top.  Left: plugin name.  Right: MIDI LED,
bypass button, PANIC button (`rt-ir-cab` only).

- **MIDI LED** (`◉ MIDI`): flashes `signal.midi.ok` on any
  recognised PC / CC event; `signal.midi.err` on unrecognised.
  Tooltip shows the last received message (`PC 3 on ch 1`).
- **Bypass**: latched toggle, big.  When engaged, the whole plugin
  window fades to 50 % opacity as a visual confirmation.
- **PANIC** (cab only): momentary.  Flushes the convolver's FDL and
  crossfade state.  Useful if upstream NaN or a huge transient
  leaked in — one click restores clean state without a session
  reload.

## `rt-ir-cab` — full layout

### Window

640 × 320 px, non-resizable in 1.0.  (Resizable in a later release
if users ask.)

### Regions

```
┌──────────────────────────── header (32) ──────────────────────────────────┐
│  rt-ir-cab                             ◉ MIDI   [ BYPASS ]   [ PANIC ]    │
├──────────────────────────── slot rack (72) ───────────────────────────────┤
│  [1 Fend] [2 Marx] [3 Mesa*] [4 V30 ] [5 Grn ] [6  · ] [7  · ] [8  · ]    │
├──────────────────────────── waveform (108) ───────────────────────────────┤
│    ▁▂▄▇█▇▆▅▄▃▂▂▁▁▁                                                       │
│    ══════════════════════════════════════════════════════════════════════ │
│    Mesa 4x12 · 512 taps · 48 kHz · 10.7 ms                                │
├──────────────────────────── controls (96) ────────────────────────────────┤
│   IN GAIN    LOW CUT    HIGH CUT    MIX     OUT GAIN    ✓ NORMALIZE       │
│    (◉)        (◉)        (◉)        (◉)      (◉)                          │
│    0 dB      80 Hz     6.5 kHz     100 %     0 dB          CROSSFADE      │
│                                                             (◉) 25 ms     │
└──────────────────────────── footer (12 padding) ──────────────────────────┘
```

Active slot marked `*`.

### Controls

| Control       | Range         | Default | Notes                          |
|---------------|---------------|---------|--------------------------------|
| In Gain       | −24…+24 dB    | 0 dB    | Pre-EQ, pre-convolution        |
| Low Cut       | 20–500 Hz     | 80 Hz   | 12 dB/oct, pre-convolution     |
| High Cut      | 2–20 kHz      | 20 kHz  | 12 dB/oct, post-convolution    |
| Mix           | 0–100 %       | 100 %   | 0 % = fully dry                |
| Out Gain      | −24…+24 dB    | 0 dB    | Post everything                |
| Normalize     | on / off      | on      | IR loudness compensation       |
| Crossfade     | 5–200 ms      | 25 ms   | Slot-to-slot transition        |

### Visual identity

Accent: **amber** (`accent.cab`).  Slight warm cast on the waveform
(`#e6c489` @ 70 % opacity) so the plugin feels valve-amp-adjacent
without leaning kitsch.

## `rt-ir-reverb` — full layout

### Window

720 × 380 px, non-resizable in 1.0.

### Regions

```
┌──────────────────────────── header (32) ──────────────────────────────────┐
│  rt-ir-reverb                                     ◉ MIDI   [ BYPASS ]     │
├──────────────────────────── slot rack (72) ───────────────────────────────┤
│  [1 Room] [2 Hall*] [3 Plat] [4 Sprg] [5  · ] [6  · ] [7  · ] [8  · ]     │
├──────────────────────────── waveform (128) ───────────────────────────────┤
│ L ▂▅█▇▆▅▄▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                                           │
│ R ▂▅█▇▆▅▄▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                                           │
│ ═════════════════════════════════════════════════════════════════════════ │
│ Concert Hall · 2.40 s · true-stereo · 48 kHz    ⇄ REVERSE  ✂ LENGTH 85 %  │
├──────────────────────────── controls (128) ───────────────────────────────┤
│  PRE-DELAY   LOW CUT   HIGH CUT   WIDTH   CROSSFADE                       │
│   (◉)         (◉)        (◉)       (◉)      (◉)                           │
│   40 ms     200 Hz     12 kHz    140 %    120 ms                          │
│                                                                           │
│                                       DRY  ▐████░░░░░░░░▌   0.0 dB        │
│                                       WET  ▐███░░░░░░░░░▌  -6.0 dB        │
└──────────────────────────── footer (12 padding) ──────────────────────────┘
```

### Controls

| Control       | Range              | Default | Notes                       |
|---------------|--------------------|---------|-----------------------------|
| Pre-Delay     | 0–500 ms           | 0 ms    | Delay before wet path       |
| Low Cut       | 20–500 Hz          | 20 Hz   | Wet only                    |
| High Cut      | 2–20 kHz           | 20 kHz  | Wet only                    |
| Width         | 0–200 %            | 100 %   | Stereo width on wet         |
| Crossfade     | 5–500 ms           | 120 ms  | Longer than cab — tails     |
| Dry (fader)   | −∞…+6 dB           | 0 dB    | Separate from Wet           |
| Wet (fader)   | −∞…+6 dB           | −6 dB   | Separate from Dry           |
| IR Reverse    | on / off           | off     | Toggle in waveform strip    |
| IR Length     | 1–100 %            | 100 %   | Truncates tail, CPU control |

### Visual identity

Accent: **teal** (`accent.reverb`).  Slight cool cast on the
waveform (`#82c9d1` @ 70 % opacity).  Feels like a studio outboard
reverb, not a guitar pedal.

## Interaction flows

### Loading an IR

1. User drops a WAV/FLAC/AIFF onto slot 3, **or** right-click →
   Load, **or** double-click.
2. Slot pad enters **Loading** state (animated dots).
3. Worker thread loads → resamples to host rate → pre-partitions.
4. On success: slot pad enters **Filled + inactive** state.  If it
   was the first fill and no other slot is active, it auto-activates.
5. On failure: slot pad reverts to **Empty**, a toast surface at the
   bottom of the window shows the error for 4 s.

### Switching slots

1. User clicks slot 2 (or sends the mapped MIDI PC).
2. Crossfade begins immediately.  Both convolvers run in parallel
   for `Crossfade` ms.
3. Slot 2 becomes the active slot (thick accent border).  Slot 3
   returns to inactive.
4. Waveform display transitions to slot 2's IR (fade, matching the
   audio crossfade duration).

### MIDI learn

1. Right-click any parameter or slot pad → **MIDI Learn**.
2. Header LED enters a "listening" state (pulsing accent color).
3. First recognised MIDI event within 10 s binds to that
   parameter/slot.  Escape or a second right-click cancels.
4. Bound MIDI mappings show as a small text tag next to the control
   (e.g. `CC #21`, `PC #2`).
5. Right-click → **Unlearn** removes the binding.

### PANIC (cab only)

1. User clicks PANIC.
2. Convolver FDL zeroed, crossfade state reset, gain smoothers
   snapped to their targets.  Denormal guard is re-armed.
3. One block of silence, then normal audio resumes.  No file reload,
   no plugin reinstantiation.

## Accessibility

- **Contrast**: all text meets WCAG AA against its background.
- **Focus**: every interactive control has a visible focus ring
  (2 px `accent.*`).  Focus order is left-to-right, top-to-bottom.
- **Keyboard**:
  - Tab / Shift-Tab moves focus.
  - Arrow keys navigate the slot rack.
  - Enter / Space activates a slot or toggles a button.
  - Up / Down on a focused knob adjusts value (Shift for fine).
- **Screen readers**: every control has an `aria-label` (or
  equivalent in the host's a11y bridge) stating name and current
  value, e.g. "Low Cut, 80 Hz".
- **Colour-blindness**: the active-slot state also uses a border
  **thickness** change, not colour alone.  MIDI LED shape changes
  between ok / error, not just colour.

## Deliberately out of scope for 1.0

- Resizable / DPI-scalable window (add if requested).
- Per-slot EQ (8 × EQ = 40 controls; global EQ is enough).
- Built-in IR editor / trimmer (DAWs already have this).
- Snapshot / patch morphing beyond the 8-slot crossfade (host
  handles preset management).
- Spectrum analyser or output metering (host provides meters).
- Multiple simultaneous IRs mixed together (out of scope of the
  underlying library primitives).

## Open questions

- **Font licensing**: Inter and JetBrains Mono are both OFL — safe
  to bundle.  Confirm no host-specific font restrictions.
- **Icon set**: currently unspecified.  Candidates: Lucide (MIT),
  Phosphor (MIT).  Pick one and stick to it.
- **Localisation**: 1.0 ships English-only.  Which languages should
  1.1 target?  (French, German, Japanese are the usual audio-plugin
  starting set.)
