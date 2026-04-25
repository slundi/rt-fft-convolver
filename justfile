set show-recipe-help := true

# --- Variables ---
binary_name := "rust_template" # Change this to your project name

# --- Default ---
[help]
default:
    @just --list

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

# Run all pre-commit hooks manually on all files
check:
    prek run --all-files

# Force update hooks
update-hooks:
    prek autoupdate

# Run gitleaks to scan for secrets
scan-secrets:
    gitleaks detect --verbose --redact

# --- Coverage ---

# Generate HTML coverage report and open it
coverage:
    cargo llvm-cov --html --open

# Generate LCOV coverage report
coverage-lcov:
    cargo llvm-cov --lcov --output-path lcov.info

# Check that newly added .rs files have >= 80% line coverage
# Usage: just coverage-check          (compares against master)
#        just coverage-check main      (compares against a different base)
coverage-check base="master":
    #!/usr/bin/env bash
    set -euo pipefail
    NEW_FILES=$(git diff --name-only --diff-filter=A "origin/{{base}}...HEAD" -- '*.rs' 2>/dev/null \
        || git diff --name-only --diff-filter=A HEAD~1 -- '*.rs' 2>/dev/null || true)
    if [ -z "$NEW_FILES" ]; then
        echo "No new Rust files detected, skipping coverage check."
        exit 0
    fi
    echo "Checking coverage for new files:"
    echo "$NEW_FILES"
    cargo llvm-cov --json --output-path /tmp/llvm-cov-report.json
    FAILED=0
    while IFS= read -r FILE; do
        [ -z "$FILE" ] && continue
        COVERAGE=$(jaq -r --arg f "$FILE" \
            '.data[0].files[] | select(.filename | endswith($f)) | .summary.lines.percent' \
            /tmp/llvm-cov-report.json | head -1)
        if [ -z "$COVERAGE" ] || [ "$COVERAGE" = "null" ]; then
            echo "⚠  No coverage data for $FILE (skipping)"
        elif awk "BEGIN { exit ($COVERAGE < 80) }"; then
            echo "✅ $FILE: ${COVERAGE}%"
        else
            echo "❌ $FILE: ${COVERAGE}% (minimum: 80%)"
            FAILED=1
        fi
    done <<< "$NEW_FILES"
    exit $FAILED

# --- Cleanup ---

# Clean build artifacts
clean:
    cargo clean

# --- CI Simulation ---

# Run the full pipeline as it would run in CI
ci: fmt lint test
    @echo "✅ All checks passed!"
