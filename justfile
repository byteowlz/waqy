# waqy: wake-phrase detection as data. Code, docs and a test fixture only;
# releases and training data live in a model store / data root outside git.

root := env_var_or_default("WAQY_ROOT", env_var("HOME") + "/.local/share/foxline/waqy-train-1")
models := env_var_or_default("WAQY_MODELS", env_var("HOME") + "/.local/share/foxline/waqy-models")

default:
    @just --list

# Rust: protocol, detector, parity (set WAKE_PARITY_JSON + WAKE_MANIFEST for parity)
check:
    cargo fmt -- --check
    cargo clippy --release -- -D warnings
    cargo test --release

# Browser engine: streaming parity and half-duplex gate
check-web:
    cd web && bun test

# Training pipeline environment and screening tests
check-training:
    cd training && uv sync
    cd training && uv run --no-project python -m pytest -q

# Address variants, collection categories and synthesis inventory (no model inference)
check-word-inventory:
    cd training && uv run --no-project python -m unittest test_address_variants -v

# Acoustic-condition reporting regression tests (synthetic fixtures; no recordings/GPU)
check-room-gate:
    cd training && uv run --no-project --with numpy --with soundfile python -m unittest test_gate_rooms -v

# Everything that does not need a GPU
check-all: check check-web check-training

# Validate a release folder (hashes, protocol, phrases)
check-release manifest:
    cargo run --release -- check --manifest {{manifest}}

# CPU cost of a release, per 80 ms frame
bench manifest:
    cargo run --release -- bench --manifest {{manifest}}

# Optional private deployment recipes; absent from the public source snapshot.
import? '.internal/dev.just'

# Shared defaults for the standalone browser UI
packs := env("WAQY_PACKS_ROOT", home_directory() / ".local/share/foxline/waqy-models")
ui_env := "WAQY_MODELS_DIR=" + models + "/spqx-r3" + " WAQY_PACKS_DIR=" + packs

# Serve the standalone UI: / records phrases, /detect tests a release
ui:
    cd web/app && {{ui_env}} bun server.mjs

# UI checks: server routes and stores (no browser), then the two browser checks
check-ui:
    cd web/app && bun test

check-ui-browser:
    @echo "needs PLAYWRIGHT, CHROMIUM, AUDIO_FIXTURE, SERVER_LOG, RECORDINGS_DIR, LIVE_DIR"
    cd web/app && bun browser-check.mjs
    cd web/app && bun detect-check.mjs

# Third-take faults in All words: local server and disposable RECORDINGS_DIR only.
check-recording-regression:
    cd web/app && bun recording-regression.mjs

# /heads opt-in recording: all heads, surgical logs, replay, marks and save-failure recovery.
check-heads-recording:
    cd web/app && bun heads-recording-check.mjs

# Cut head-recording review candidates; extra flags include --out and --marked-only.
cut-head-recording session *args:
    bun tools/cut-head-recording.mjs --session {{quote(session)}} {{args}}

# /detect picker: a published pack, a subset of its heads, live scores.
check-heads-picker:
    cd web/app && bun heads-picker-check.mjs

# Multi-mic capture: two synthetic streams, one started late, plus a broken device.
check-multi-mic:
    cd web/app && bun multi-mic-check.mjs

# Look-alike editor end to end: local server, disposable RECORDINGS_DIR and LOOKALIKES_FILE only.
check-lookalikes:
    cd web/app && bun lookalikes-check.mjs

# Head explorer viewport and fake-microphone checks; external browser/fixture paths are explicit.
check-heads-browser:
    @echo "needs PLAYWRIGHT, CHROMIUM, POSITIVE_WAV, TEST_ORIGIN, EVIDENCE_DIR"
    cd web/app && bun heads-check.mjs

# Serve a release over wake-stream/1 (hosts spawn this)
listen manifest:
    cargo run --release -- listen --manifest {{manifest}}

# Package an immutable release from trained heads
release name +heads:
    cd training && uv run --no-project python release.py {{root}} {{name}} {{heads}}

# Offline, read-only preview of an explicit public subset. Extra flags: --metadata/--consent/--catalog.
[positional-arguments]
publish-heads source name heads *args:
    #!/usr/bin/env bash
    set -euo pipefail
    source=$1; name=$2; heads=$3; shift 3
    exec uv run --no-project python training/publish_hf.py "$source" --name "$name" --heads "$heads" "$@"

# Explicit model publication, only after approving the preview and any owner-derived weights.
[positional-arguments]
publish-heads-apply source name heads *args:
    #!/usr/bin/env bash
    set -euo pipefail
    source=$1; name=$2; heads=$3; shift 3
    exec uv run --no-project --with huggingface-hub==0.36.0 python training/publish_hf.py "$source" --name "$name" --heads "$heads" --apply "$@"

# Optional private TOML profile; CLI flags still override it and default to preview.
[positional-arguments]
publish-heads-config config *args:
    #!/usr/bin/env bash
    set -euo pipefail
    config=$1; shift
    exec uv run --no-project python training/publish_hf.py --config "$config" "$@"

# Applying a configured selection remains an explicit operation, never a TOML setting.
[positional-arguments]
publish-heads-config-apply config *args:
    #!/usr/bin/env bash
    set -euo pipefail
    config=$1; shift
    exec uv run --no-project --with huggingface-hub==0.36.0 python training/publish_hf.py --config "$config" --apply "$@"

# Generated-byte fixtures and a fake HF client; no credentials, models or network.
check-publication:
    cd training && uv run --no-project python -m unittest -v test_publication test_head_export_config

# Training pipeline, by stage (see training/README.md)
synth model n_train="7000" n_test="1200":
    cd training && uv run --no-project python build_data.py {{root}} synth {{model}} {{n_train}} {{n_test}}
    cd training && uv run --no-project python build_data.py {{root}} asrgate {{model}}

negatives:
    cd training && uv run --no-project python build_data.py {{root}} stream musan
    cd training && uv run --no-project python build_data.py {{root}} stream voxpopuli
    cd training && uv run --no-project python build_data.py {{root}} validation 40
    cd training && uv run --no-project python build_data.py {{root}} negatives 400

train model:
    cd training && uv run --no-project python pipeline.py {{root}} {{model}} config
    cd training && WAKE_DEVICE=cuda uv run --no-project python pipeline.py {{root}} {{model}} train
    cd training && uv run --no-project python pipeline.py {{root}} {{model}} export
