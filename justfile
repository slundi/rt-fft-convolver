# rt-fft-convolver / rt-ir plugins — task runner.
#
# Recipes mirror what CI runs so the same checks work locally
# (roadmap Phase 5: "CI jobs must also be runnable with `just`").

# --- Variables ---

# Plugin packages that produce CLAP/VST3 bundles.
plugins := "rt-ir-cab rt-ir-reverb"
# Feature set for a shipping build (includes the egui editor → needs X11/OpenGL).
gui_features := "gui"

# --- Development ---

# Build the default members in debug mode (framework-agnostic, no plugin UI).
build:
    cargo build

# Watch for changes and re-run the tests (requires cargo-watch).
watch:
    cargo watch -x test

# --- Quality Control ---

# Run all tests (unit + integration).
test:
    cargo test

# Full health check (vulnerabilities, unused deps, licenses).
health-check:
    cargo audit
    cargo machete
    cargo deny check

# Clippy on the framework-agnostic build — no system libraries needed.
lint:
    cargo clippy --all-targets -- -D warnings

# Clippy the egui editors (needs the X11/OpenGL dev libraries).
lint-gui:
    cargo clippy -p rt-ir-cab --features {{gui_features}} -- -D warnings
    cargo clippy -p rt-ir-reverb --features {{gui_features}} -- -D warnings

# Format all code.
fmt:
    cargo fmt --all
    dprint fmt
    find . -name '*.nix' -not -path './target/*' | xargs --no-run-if-empty nixfmt
    yamlfmt -exclude "target/**" .

# Fail if anything is unformatted (CI gate).
fmt-check:
    cargo fmt --all --check

# Run all pre-commit hooks manually on all files.
check:
    prek run --all-files

# Scan for secrets.
scan-secrets:
    gitleaks detect --verbose --redact

# Generate lcov.info for editor coverage display.
coverage:
    cargo llvm-cov --lcov --output-path lcov.info

# Enforce the 80% line-coverage threshold. BASE is accepted for CI compatibility.
coverage-check BASE="master":
    cargo llvm-cov --lcov --output-path lcov.info
    cargo llvm-cov report --fail-under-lines 80

# --- Bundling (CLAP + VST3 via NIH-plug's xtask) ---

# Bundle one plugin with the editor. Usage: just bundle rt-ir-cab
bundle plugin:
    cargo xtask bundle {{plugin}} --release --features {{gui_features}}

# Bundle every plugin with the editor (release; needs X11/OpenGL).
bundle-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for p in {{plugins}}; do
        cargo xtask bundle "$p" --release --features {{gui_features}}
    done

# Bundle every plugin headless (no editor) — no system libraries needed.
bundle-headless:
    #!/usr/bin/env bash
    set -euo pipefail
    for p in {{plugins}}; do
        cargo xtask bundle "$p" --release --features nih
    done

# Bundle a plugin for a cross target (needs `cross` + the target's X11/GL libs).
# Usage: just bundle-cross rt-ir-cab aarch64-unknown-linux-gnu
bundle-cross plugin target:
    cargo xtask bundle {{plugin}} --release --features {{gui_features}} --target {{target}}

# --- Packaging ---

# SHA-256 checksums of every bundle → target/bundled/SHA256SUMS.
checksums:
    packaging/checksums.sh

# Build a Debian package from the bundles in target/bundled.
# Usage: just deb 0.1.0
deb version="0.1.0":
    packaging/build-deb.sh "{{version}}"

# Stage a per-platform release archive (+ deb on Linux) into dist/.
# Usage: just package 0.1.0
package version="0.1.0":
    packaging/package.sh "{{version}}"

# --- CI ---

# The default CI gate (framework-agnostic; no system libraries).
ci: fmt-check lint test
    @echo "Core checks passed!"

# The GUI CI gate (run where X11/OpenGL is available).
ci-gui: lint-gui
    cargo build -p rt-ir-cab --features {{gui_features}}
    cargo build -p rt-ir-reverb --features {{gui_features}}

# The full pipeline (use inside `nix develop`).
ci-all: fmt lint test health-check scan-secrets

# Run the full pipeline through the Nix devshell — no docker/podman needed.
ci-local:
    nix develop --command just fmt-check
    nix develop --command just lint
    nix develop --command just test
    nix develop --command just health-check
    nix develop --command just scan-secrets
    @echo "All CI checks passed!"

# --- Changelog & release ---

# Generate or update CHANGELOG.md from conventional commits.
changelog:
    git cliff -o CHANGELOG.md
    @echo "CHANGELOG.md updated — commit it before tagging"

# Preview changelog entries not yet in a release.
changelog-unreleased:
    git cliff --unreleased

# Create and push an annotated release tag, triggering the release workflow.
# Usage: just tag v1.2.3
tag VERSION:
    git tag -a {{VERSION}} -m "Release {{VERSION}}"
    git push origin {{VERSION}}
    @echo "Pushed tag {{VERSION}} — the release workflow will start on Codeberg"

# Package native bundles and publish a release to Codeberg via the Forgejo API.
# Requires: CODEBERG_TOKEN, xh, jq. Cross-OS artifacts come from the CI matrix;
# this publishes whatever is staged in dist/ (defaults to the host platform).
# Usage: just release v1.2.3
release VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${CODEBERG_TOKEN:?Set CODEBERG_TOKEN to a Codeberg API token}"

    remote=$(git remote get-url origin)
    repo_path="${remote#https://codeberg.org/}"
    repo_path="${repo_path#git@codeberg.org:}"
    repo_path="${repo_path%.git}"

    version_no_v="${VERSION#v}"
    echo "==> Packaging {{VERSION}} for the host platform..."
    packaging/package.sh "$version_no_v"

    prerelease=false
    echo "{{VERSION}}" | grep -qE '(alpha|beta|rc)' && prerelease=true || true
    notes=$(git cliff --unreleased --strip all 2>/dev/null || true)

    echo "==> Creating Codeberg release..."
    release_id=$(xh -A bearer -a "$CODEBERG_TOKEN" \
        POST "https://codeberg.org/api/v1/repos/${repo_path}/releases" \
        tag_name="{{VERSION}}" \
        name="{{VERSION}}" \
        body="$notes" \
        draft:=false \
        prerelease:=$prerelease \
        | jq '.id')

    echo "==> Uploading assets (release id: ${release_id})..."
    for f in dist/*; do
        [ -f "$f" ] || continue
        echo "  -> $(basename "$f")"
        xh -A bearer -a "$CODEBERG_TOKEN" -f \
            POST "https://codeberg.org/api/v1/repos/${repo_path}/releases/${release_id}/assets" \
            "attachment@${f}"
    done

    echo ""
    echo "Released: https://codeberg.org/${repo_path}/releases/tag/{{VERSION}}"

# --- Cleanup ---

# Remove build artifacts.
clean:
    cargo clean
    rm -rf dist
