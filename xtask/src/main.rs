//! `cargo xtask` entry point — delegates to NIH-plug's bundler.
//!
//! Usage:
//!
//! ```text
//! cargo xtask bundle rt-ir-cab --release --features gui
//! cargo xtask bundle rt-ir-reverb --release --features gui
//! ```
//!
//! Bundles land in `target/bundled/` as `<name>.clap` and `<name>.vst3`.

fn main() -> nih_plug_xtask::Result<()> {
    nih_plug_xtask::main()
}
