//! `PreparedModel` — the RT-safe entry point uniting LSTM and WaveNet.
//!
//! # What the loader produces
//!
//! An off-thread loader (NAM `.nam` or Proteus / GuitarML `.json`) parses
//! a model file, validates it, transposes weight tensors into the
//! gate-major / channel-major layouts expected by [`LstmCell`] and
//! [`Conv1d`], and hands the result to the audio thread as a
//! [`PreparedModel`] enum.
//!
//! The audio thread then calls the single hot method:
//!
//! ```rust,ignore
//! # use rt_neural_core::PreparedModel;
//! let output_sample = model.process_sample(input_sample);
//! ```
//!
//! # Architecture dispatch
//!
//! The enum uses static dispatch — the audio thread pays one branch per
//! sample to select LSTM vs WaveNet.  Once inside the arm, all further
//! calls are direct.  This mirrors how NAM's C++ engine dispatches, and
//! avoids the vtable indirection of a trait object.
//!
//! # Real-time safety
//!
//! Both [`process_sample`](PreparedModel::process_sample) and
//! [`reset`](PreparedModel::reset) are allocation-free.  A future
//! [`slot_bank`](crate::slot_bank) layer handles model *swap* off-thread.

use crate::conv1d::Conv1d;
use crate::lstm::LstmCell;
use crate::metadata::ModelMetadata;

// ─── LSTM model ──────────────────────────────────────────────────────────────

/// A stacked LSTM amp model — one or more [`LstmCell`] layers followed by a
/// dense head that projects to a single output sample.
///
/// This is the shape NAM's LSTM variants and every Proteus / GuitarML
/// model take: `N` recurrent layers of identical hidden width, then a
/// `[hidden_size, 1]` linear head.
pub struct LstmModel {
    /// Recurrent layers; each consumes the previous layer's hidden state.
    /// The first layer takes the raw audio sample (input width 1).
    layers: Vec<LstmCell>,
    /// Dense head, shape `[1, hidden_size]`, applied to the final layer's
    /// hidden state.  Stored as a plain slice so a `Dense` isn't overkill
    /// for a 1-row matmul.
    head_weights: Box<[f32]>,
    /// Scalar bias of the head.
    head_bias: f32,
    /// Scratch buffer of length 1 (`process_sample` returns f32 via this).
    output_scratch: [f32; 1],
    /// Provenance / training metadata (unknown by default; populated by
    /// the loader or [`with_metadata`](Self::with_metadata)).
    metadata: ModelMetadata,
}

impl LstmModel {
    /// Construct a stacked LSTM amp model.
    ///
    /// `layers` must be non-empty.  The final layer's hidden width must
    /// match `head_weights.len()`.  For NAM/Proteus models the input to
    /// the *first* layer is a single audio sample, so
    /// `layers[0].input_size() == 1`.
    pub fn new(layers: Vec<LstmCell>, head_weights: Vec<f32>, head_bias: f32) -> Self {
        debug_assert!(!layers.is_empty(), "LstmModel: layers must be non-empty");
        debug_assert_eq!(
            layers.last().expect("at least one layer").hidden_size(),
            head_weights.len(),
            "LstmModel: head_weights.len() must equal final layer hidden size"
        );
        Self {
            layers,
            head_weights: head_weights.into_boxed_slice(),
            head_bias,
            output_scratch: [0.0f32; 1],
            metadata: ModelMetadata::default(),
        }
    }

    /// Attach provenance / training metadata to the model.  Chainable —
    /// callers typically do
    /// `LstmModel::new(..).with_metadata(loader_metadata)`.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Read-only view of the attached metadata.
    #[inline]
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Total number of stored scalar parameters across the LSTM stack and
    /// the dense head (`+1` for `head_bias`).
    pub fn parameter_count(&self) -> usize {
        let layers_count: usize = self.layers.iter().map(LstmCell::parameter_count).sum();
        layers_count + self.head_weights.len() + 1
    }

    /// Number of stacked LSTM layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden width shared by all layers (LSTM stacks in NAM / Proteus
    /// carry a single hidden width across the stack).
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.layers.first().map(LstmCell::hidden_size).unwrap_or(0)
    }

    /// Zero every layer's recurrent state.
    #[inline]
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    /// Process one audio sample.
    #[inline]
    fn process_sample(&mut self, input_sample: f32) -> f32 {
        self.output_scratch[0] = input_sample;
        // Feed each recurrent layer with the previous layer's hidden output.
        // Layer 0 takes the raw sample; subsequent layers consume the
        // previous layer's `hidden()` slice.
        let mut input_view: &[f32] = &self.output_scratch;
        for layer in self.layers.iter_mut() {
            layer.process_sample(input_view);
            input_view = layer.hidden();
        }

        // Head: dot(head_weights, final_hidden) + head_bias.
        let final_hidden = self
            .layers
            .last()
            .expect("LstmModel: layers must be non-empty")
            .hidden();
        let mut accumulator = self.head_bias;
        for (weight, hidden_value) in self.head_weights.iter().zip(final_hidden.iter()) {
            accumulator += weight * hidden_value;
        }
        accumulator
    }
}

// ─── WaveNet model ───────────────────────────────────────────────────────────

/// A residual block inside a WaveNet stack.
///
/// A block is one dilated causal conv followed by a `1×1` projection and a
/// residual add.  For inference we keep only what the NAM WaveNet uses:
/// a single-tap gate on the residual, no separate skip pathway.
///
/// This is the minimum footprint that keeps NAM "Feather" and "Standard"
/// models round-trippable; richer WaveNet variants (Nam LightBox, custom
/// research models) can extend this struct in a later phase without
/// changing the enum surface.
pub struct WaveNetBlock {
    /// The dilated causal 1-D convolution.  Its `out_channels` equals the
    /// block's channel count.
    conv: Conv1d,
    /// 1×1 projection back to the residual channel count — trivial dense
    /// applied per sample.
    projection_weights: Box<[f32]>,
    /// Per-channel bias for the projection.
    projection_bias: Box<[f32]>,
    /// Scratch buffer sized to `channels` for the conv output.
    conv_scratch: Box<[f32]>,
    channels: usize,
}

impl WaveNetBlock {
    /// Assemble one residual block.
    ///
    /// - `conv` — the dilated causal Conv1D producing `channels` outputs.
    /// - `projection_weights` — length `channels · channels`, row-major.
    /// - `projection_bias` — length `channels`.
    ///
    /// **Not** real-time safe.
    pub fn new(conv: Conv1d, projection_weights: Vec<f32>, projection_bias: Vec<f32>) -> Self {
        let channels = conv.out_channels();
        debug_assert_eq!(projection_weights.len(), channels * channels);
        debug_assert_eq!(projection_bias.len(), channels);
        Self {
            conv,
            projection_weights: projection_weights.into_boxed_slice(),
            projection_bias: projection_bias.into_boxed_slice(),
            conv_scratch: vec![0.0f32; channels].into_boxed_slice(),
            channels,
        }
    }

    /// Number of residual channels.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Total number of stored scalar parameters (dilated conv + 1×1
    /// projection weights + projection bias).
    #[inline]
    pub fn parameter_count(&self) -> usize {
        self.conv.parameter_count() + self.projection_weights.len() + self.projection_bias.len()
    }

    /// Zero the internal conv1d history.
    #[inline]
    fn reset(&mut self) {
        self.conv.reset();
    }

    /// Consume one input frame (length `channels`) and return the residual
    /// output (in place — `frame` is mutated so the caller can chain blocks
    /// without allocating).
    #[inline]
    fn process_frame(&mut self, frame: &mut [f32]) {
        debug_assert_eq!(frame.len(), self.channels);

        // Dilated causal conv on the incoming frame.
        self.conv.forward(frame, &mut self.conv_scratch);

        // 1×1 projection + residual add.  Layout: projection_weights is
        // row-major `[channels, channels]`.
        for (row_index, frame_cell) in frame.iter_mut().enumerate() {
            let row_start = row_index * self.channels;
            let row = &self.projection_weights[row_start..row_start + self.channels];
            let mut accumulator = self.projection_bias[row_index];
            for (weight, conv_output) in row.iter().zip(self.conv_scratch.iter()) {
                accumulator += weight * conv_output;
            }
            // Residual: y = x + projection(conv(x)).
            *frame_cell += accumulator;
        }
    }
}

/// A WaveNet-style amp model — a lift-in dense that broadens the mono
/// sample to `channels` features, a stack of dilated residual blocks, and a
/// project-out dense that collapses back to a mono sample.
///
/// The dilations grow geometrically (typically `1, 2, 4, ..., 512`), which
/// gives the receptive field required to capture guitar harmonic decay
/// with a much smaller parameter budget than an equivalent LSTM would need
/// for the same fidelity.
pub struct WaveNetModel {
    /// Lift-in: shape `[channels, 1]`.  Flat storage.
    lift_weights: Box<[f32]>,
    /// Lift-in bias, length `channels`.
    lift_bias: Box<[f32]>,
    /// Residual blocks (stack ordered by depth).
    blocks: Vec<WaveNetBlock>,
    /// Project-out: shape `[1, channels]`.  Flat storage.
    projection_weights: Box<[f32]>,
    /// Project-out scalar bias.
    projection_bias: f32,
    /// Scratch buffer sized to `channels`, reused every sample.
    frame_scratch: Box<[f32]>,
    channels: usize,
    /// Provenance / training metadata (unknown by default; populated by
    /// the loader or [`with_metadata`](Self::with_metadata)).
    metadata: ModelMetadata,
}

impl WaveNetModel {
    /// Assemble a WaveNet amp model.
    ///
    /// All `blocks` must share the same `channels` count and match
    /// `lift_weights.len() == channels` and `projection_weights.len() ==
    /// channels`.
    pub fn new(
        lift_weights: Vec<f32>,
        lift_bias: Vec<f32>,
        blocks: Vec<WaveNetBlock>,
        projection_weights: Vec<f32>,
        projection_bias: f32,
    ) -> Self {
        debug_assert!(!blocks.is_empty(), "WaveNetModel: blocks must be non-empty");
        let channels = blocks[0].channels();
        debug_assert!(
            blocks.iter().all(|block| block.channels() == channels),
            "WaveNetModel: all blocks must share the same channel count"
        );
        debug_assert_eq!(lift_weights.len(), channels);
        debug_assert_eq!(lift_bias.len(), channels);
        debug_assert_eq!(projection_weights.len(), channels);
        Self {
            lift_weights: lift_weights.into_boxed_slice(),
            lift_bias: lift_bias.into_boxed_slice(),
            blocks,
            projection_weights: projection_weights.into_boxed_slice(),
            projection_bias,
            frame_scratch: vec![0.0f32; channels].into_boxed_slice(),
            channels,
            metadata: ModelMetadata::default(),
        }
    }

    /// Attach provenance / training metadata to the model.  Chainable.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Read-only view of the attached metadata.
    #[inline]
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Total number of stored scalar parameters (lift-in weights + bias,
    /// every block's parameters, project-out weights, and the scalar
    /// projection bias).
    pub fn parameter_count(&self) -> usize {
        let blocks_count: usize = self.blocks.iter().map(WaveNetBlock::parameter_count).sum();
        self.lift_weights.len()
            + self.lift_bias.len()
            + blocks_count
            + self.projection_weights.len()
            + 1
    }

    /// Number of residual blocks in the stack.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Zero every block's history.
    #[inline]
    pub fn reset(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset();
        }
    }

    /// Process one audio sample.
    #[inline]
    fn process_sample(&mut self, input_sample: f32) -> f32 {
        // ── Lift-in: broadcast the mono sample to `channels` features.
        for (channel_index, frame_cell) in self.frame_scratch.iter_mut().enumerate() {
            *frame_cell =
                self.lift_weights[channel_index] * input_sample + self.lift_bias[channel_index];
        }

        // ── Stack of residual blocks; each mutates the frame in-place.
        for block in self.blocks.iter_mut() {
            block.process_frame(&mut self.frame_scratch);
        }

        // ── Project-out: dot(projection, frame) + bias.
        let mut accumulator = self.projection_bias;
        for (weight, feature) in self
            .projection_weights
            .iter()
            .zip(self.frame_scratch.iter())
        {
            accumulator += weight * feature;
        }
        accumulator
    }

    /// Number of internal residual channels (also called "block width").
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

// ─── The dispatch enum ───────────────────────────────────────────────────────

/// A ready-to-run neural amp model.
///
/// Constructed by the (Phase-2) NAM / Proteus loader, then owned by an
/// [`crate::slot_bank::ModelSlotBank`] for click-free hot-swap.  Cheap to
/// hold in an `Option<PreparedModel>` — the empty slot is a null variant.
pub enum PreparedModel {
    /// Stacked LSTM (NAM LSTM variant, Proteus / GuitarML).
    Lstm(LstmModel),
    /// Stacked WaveNet residual blocks (NAM "Feather" / "Standard").
    WaveNet(WaveNetModel),
}

impl PreparedModel {
    /// Process one audio sample.
    ///
    /// **Real-time safe** — one branch on the variant tag, then a straight
    /// call into the concrete model.
    #[inline]
    pub fn process_sample(&mut self, input_sample: f32) -> f32 {
        match self {
            PreparedModel::Lstm(model) => model.process_sample(input_sample),
            PreparedModel::WaveNet(model) => model.process_sample(input_sample),
        }
    }

    /// Zero all recurrent state / ring buffers.
    #[inline]
    pub fn reset(&mut self) {
        match self {
            PreparedModel::Lstm(model) => model.reset(),
            PreparedModel::WaveNet(model) => model.reset(),
        }
    }

    /// Read-only view of the provenance / training metadata carried with
    /// this model.  Used by the plugin editor's model-info panel.
    #[inline]
    pub fn metadata(&self) -> &ModelMetadata {
        match self {
            PreparedModel::Lstm(model) => model.metadata(),
            PreparedModel::WaveNet(model) => model.metadata(),
        }
    }

    /// Human-readable name of the architecture — used by the editor's
    /// metadata panel.
    #[inline]
    pub fn architecture_name(&self) -> &'static str {
        match self {
            PreparedModel::Lstm(_) => "LSTM",
            PreparedModel::WaveNet(_) => "WaveNet",
        }
    }

    /// Total number of stored scalar parameters in the model.
    #[inline]
    pub fn parameter_count(&self) -> usize {
        match self {
            PreparedModel::Lstm(model) => model.parameter_count(),
            PreparedModel::WaveNet(model) => model.parameter_count(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ModelError;

    // ── LSTM model wrapper ───────────────────────────────────────────────────

    /// An all-zero LSTM stack plus zero head should emit exactly 0.
    #[test]
    fn zero_lstm_stack_emits_zero() -> Result<(), ModelError> {
        let hidden_size = 4;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size * input_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )?;
        let mut prepared =
            PreparedModel::Lstm(LstmModel::new(vec![cell], vec![0.0f32; hidden_size], 0.0));
        for step in 0..8 {
            let output = prepared.process_sample(step as f32);
            assert_eq!(output, 0.0);
        }
        Ok(())
    }

    /// LSTM head bias flows through when the recurrent path is silent.
    #[test]
    fn lstm_head_bias_flows_through_when_state_zero() -> Result<(), ModelError> {
        let hidden_size = 2;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size * input_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )?;
        // Freshly constructed cell has h = 0, so head output = head_bias.
        let mut model = LstmModel::new(vec![cell], vec![1.0f32, 1.0], 0.42);
        let output = model.process_sample(0.0);
        assert!((output - 0.42).abs() < 1e-6);
        Ok(())
    }

    // ── WaveNet model wrapper ────────────────────────────────────────────────

    /// A "silent" WaveNet — all weights zero, all biases zero, block
    /// projection zero — must emit `projection_bias` for every sample.
    #[test]
    fn silent_wavenet_emits_projection_bias() {
        let channels = 4;
        let conv = Conv1d::new(
            vec![0.0f32; channels * channels],
            vec![0.0f32; channels],
            channels,
            channels,
            1,
            1,
        )
        .expect("conv build");
        let block = WaveNetBlock::new(
            conv,
            vec![0.0f32; channels * channels],
            vec![0.0f32; channels],
        );
        let model = WaveNetModel::new(
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![block],
            vec![0.0f32; channels],
            0.25,
        );
        let mut prepared = PreparedModel::WaveNet(model);
        for step in 0..8 {
            let output = prepared.process_sample(step as f32);
            assert!((output - 0.25).abs() < 1e-6);
        }
    }

    /// Parameter counts sum weights + biases across every stored layer.
    #[test]
    fn parameter_count_matches_manual_sum() -> Result<(), ModelError> {
        // LSTM: 1 layer, hidden=4, input=1.
        // W_ih  = 4·4·1 = 16 weights, bias = 4·4 = 16.
        // W_hh  = 4·4·4 = 64 weights, bias = 4·4 = 16.
        // Head weights = 4, head bias = 1.
        // Total = 16 + 16 + 64 + 16 + 4 + 1 = 117.
        let hidden_size = 4;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size * input_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )?;
        let lstm = PreparedModel::Lstm(LstmModel::new(vec![cell], vec![0.0f32; hidden_size], 0.0));
        assert_eq!(lstm.parameter_count(), 117);
        assert_eq!(lstm.architecture_name(), "LSTM");

        // WaveNet: channels=3, 1 block, kernel_size=1.
        // Conv1d weights = 3·1·3 = 9, bias = 3.
        // Projection weights = 3·3 = 9, projection bias = 3.
        // Lift weights = 3, lift bias = 3.
        // Project-out weights = 3, projection scalar bias = 1.
        // Total = 9 + 3 + 9 + 3 + 3 + 3 + 3 + 1 = 34.
        let channels = 3;
        let conv = Conv1d::new(
            vec![0.0f32; channels * channels],
            vec![0.0f32; channels],
            channels,
            channels,
            1,
            1,
        )?;
        let block = WaveNetBlock::new(
            conv,
            vec![0.0f32; channels * channels],
            vec![0.0f32; channels],
        );
        let wavenet = PreparedModel::WaveNet(WaveNetModel::new(
            vec![0.0f32; channels],
            vec![0.0f32; channels],
            vec![block],
            vec![0.0f32; channels],
            0.0,
        ));
        assert_eq!(wavenet.parameter_count(), 34);
        assert_eq!(wavenet.architecture_name(), "WaveNet");
        Ok(())
    }

    /// Attaching metadata via `with_metadata` and reading it back through
    /// the enum accessor round-trips cleanly.
    #[test]
    fn metadata_round_trips_through_prepared_model() -> Result<(), ModelError> {
        use crate::metadata::ModelMetadata;

        let hidden_size = 2;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            1,
            hidden_size,
        )?;
        let meta = ModelMetadata {
            source_gear_name: Some("Fender Twin".into()),
            training_sample_rate: Some(48_000),
            training_input_level_dbfs: Some(-20.0),
            format_name: Some("NAM".into()),
        };
        let lstm =
            LstmModel::new(vec![cell], vec![0.0f32; hidden_size], 0.0).with_metadata(meta.clone());
        let prepared = PreparedModel::Lstm(lstm);

        let read = prepared.metadata();
        assert_eq!(read.source_gear_name.as_deref(), Some("Fender Twin"));
        assert_eq!(read.training_sample_rate, Some(48_000));
        assert!((read.sweet_spot_dbfs() - (-20.0)).abs() < f32::EPSILON);
        assert_eq!(read.format_name.as_deref(), Some("NAM"));
        Ok(())
    }

    /// `reset` on the enum dispatches correctly to each variant.
    #[test]
    fn reset_dispatches_to_variant() -> Result<(), ModelError> {
        // LSTM: warm up, reset, verify state cleared.
        let hidden_size = 2;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.5f32; 4 * hidden_size * input_size],
            vec![0.5f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )?;
        let mut prepared = PreparedModel::Lstm(LstmModel::new(vec![cell], vec![1.0f32, 1.0], 0.0));
        for step in 0..8 {
            let _ = prepared.process_sample(step as f32);
        }
        prepared.reset();
        // After reset, output for input=0 must equal head_bias (0 here).
        assert!(prepared.process_sample(0.0).abs() < 1e-6);
        Ok(())
    }
}
