# --- Variables ---
binary_name := "MY_PROJECT" # Change this to your project name

# --- Development ---

# Build the project in debug mode
build:
    cargo build

# Run the project
run *args:
    cargo run -- {{args}}

# Watch for changes and run (requires cargo-watch)
watch:
    cargo watch -x run

# --- Quality Control ---

# Run all tests
test:
    cargo nextest run

# Run a full health check (Vulnerabilities, Unused Deps, Licenses)
health-check:
    cargo audit
    cargo machete
    cargo deny check

# Run clippy with strict warnings
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Format all code
fmt:
    cargo fmt --all
    dprint fmt
    find . -name '*.nix' -not -path './target/*' | xargs --no-run-if-empty nixfmt
    yamlfmt -exclude "target/**" .

# Run all pre-commit hooks manually on all files
check:
    prek run --all-files

# Force update hooks
update-hooks:
    prek autoupdate

# Run gitleaks to scan for secrets
scan-secrets:
    gitleaks detect --verbose --redact

# Generate lcov.info for editor coverage display (vscode-coverage-gutters)
coverage:
    cargo llvm-cov --lcov --output-path lcov.info

# Check that overall line coverage meets the 80% threshold
# BASE is accepted for interface compatibility with CI (e.g.: just coverage-check master)
coverage-check BASE="master":
    cargo llvm-cov --lcov --output-path lcov.info
    cargo llvm-cov report --fail-under-lines 80

# --- Cleanup ---

# Clean build artifacts
clean:
    cargo clean

# --- CI ---

# Run the complete CI pipeline (use this when already inside `nix develop`)
ci-all: fmt lint test health-check scan-secrets

# Run the complete CI pipeline via Nix devshell — no docker/podman needed
# Equivalent to what Woodpecker and Forgejo Actions run in containers
ci-local:
    nix develop --command just fmt
    nix develop --command just lint
    nix develop --command just test
    nix develop --command just health-check
    nix develop --command just scan-secrets
    @echo "All CI checks passed!"

# Run a quick subset (format + lint + test) — fast feedback loop
ci: fmt lint test
    @echo "Quick checks passed!"

# --- Changelog ---

# Generate or update CHANGELOG.md from conventional commits
changelog:
    git cliff -o CHANGELOG.md
    @echo "CHANGELOG.md updated — commit it before tagging"

# Preview changelog entries for commits not yet in a release
changelog-unreleased:
    git cliff --unreleased

# --- Release ---

# Create and push an annotated release tag, triggering the release workflow
# Usage: just tag v1.2.3
tag VERSION:
    git tag -a {{VERSION}} -m "Release {{VERSION}}"
    git push origin {{VERSION}}
    @echo "Pushed tag {{VERSION}} — release workflow will start on Codeberg"

# Build a release binary for the current platform
build-release:
    cargo build --release

# Build a release binary for a specific cross-compilation target
# Requires `cross` in your PATH (included in flake.nix devShell)
# Usage: just build-release-cross aarch64-unknown-linux-gnu
build-release-cross TARGET:
    cross build --release --target {{TARGET}}

# Cross-compile, package and publish a release to Codeberg via the Forgejo API.
# Requires: CODEBERG_TOKEN env var, cross (needs Docker or Podman), xh, jq, zip
# macOS targets are skipped — cross-compilation from Linux requires the macOS SDK.
# Usage: just release v1.2.3
release VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${CODEBERG_TOKEN:?Set CODEBERG_TOKEN to a Codeberg API token (codeberg.org/user/settings/applications)}"

    remote=$(git remote get-url origin)
    repo_path="${remote#https://codeberg.org/}"
    repo_path="${repo_path#git@codeberg.org:}"
    repo_path="${repo_path%.git}"

    echo "==> Building {{VERSION}} for all targets..."
    cargo build --release --target x86_64-unknown-linux-gnu
    cross build --release --target aarch64-unknown-linux-gnu
    cross build --release --target armv7-unknown-linux-gnueabihf
    cross build --release --target x86_64-pc-windows-gnu

    echo "==> Packaging..."
    rm -rf dist && mkdir dist
    for target in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu armv7-unknown-linux-gnueabihf; do
        tar czvf "dist/{{binary_name}}-${target}.tar.gz" \
            -C "target/${target}/release" {{binary_name}}
    done
    zip "dist/{{binary_name}}-x86_64-pc-windows-gnu.zip" \
        "target/x86_64-pc-windows-gnu/release/{{binary_name}}.exe"

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
        echo "  -> $(basename "$f")"
        xh -A bearer -a "$CODEBERG_TOKEN" -f \
            POST "https://codeberg.org/api/v1/repos/${repo_path}/releases/${release_id}/assets" \
            "attachment@${f}"
    done

    echo ""
    echo "Released: https://codeberg.org/${repo_path}/releases/tag/{{VERSION}}"
