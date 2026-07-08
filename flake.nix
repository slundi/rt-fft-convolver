# When adding new tools to this template, always check the official registry to
# find the correct attribute name:
# * Search Tool: NixOS Package Search
# * Usage: If you find ripgrep, simply add pkgs.ripgrep to the buildInputs in your flake.nix.
{
  description = "Rust Template with automatic pre-commit hooks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {inherit system overlays;};
      extensions = with pkgs.vscode-extensions; [
        rust-lang.rust-analyzer
        tamasfe.even-better-toml
        jnoortheen.nix-ide
        mkhl.direnv
        vadimcn.vscode-lldb
        redhat.vscode-yaml
      ];

      # Define the rust toolchain from your toml
      rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rustToolchain
          cargo-dist
          cargo-nextest
          cargo-machete
          cargo-audit
          cargo-deny
          cargo-llvm-cov # Coverage instrumentation via LLVM
          grcov # Coverage report formatter (lcov, html, cobertura)
          jaq # JSON processor (used by coverage-check)
          prek # pre-commit
          gitleaks # The compiled secret scanner or trufflehog or ripgrep
          just # Command runner
        ];

        # This runs when the shell starts
        shellHook = ''
          echo "🦀 Rust Dev Shell Loaded"

          # Enrich the system-installed VSCodium with our extensions,
          # only adding those not already present in the user's profile.
          if command -v codium >/dev/null 2>&1; then
            ext_dir="$HOME/.vscode-oss/extensions"
            mkdir -p "$ext_dir"
            installed=$(codium --list-extensions 2>/dev/null | tr '[:upper:]' '[:lower:]')
            added=0
            for ext_pkg in ${pkgs.lib.concatMapStringsSep " " (e: "${e}") extensions}; do
              for src in "$ext_pkg"/share/vscode/extensions/*/; do
                [ -d "$src" ] || continue
                name=$(basename "$src")
                lower=$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')
                if ! printf '%s\n' "$installed" | grep -qxF "$lower"; then
                  cp -rL --no-preserve=mode,ownership "$src" "$ext_dir/$name"
                  echo "  + added VSCodium extension: $name"
                  added=$((added + 1))
                fi
              done
            done
            if [ "$added" -eq 0 ]; then
              echo "Tip: Run 'codium .' — all flake-provided extensions already installed."
            else
              echo "Tip: Restart VSCodium to pick up $added new extension(s)."
            fi
          else
            echo "Note: 'codium' not found on PATH — skipping extension seeding."
          fi

          # Automatically install hooks if .git exists
          if [ -d .git ] && command -v prek >/dev/null; then
            prek install
          fi
        '';
      };
    });
}
