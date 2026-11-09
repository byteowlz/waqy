# Training pipeline

Turns a phrase into a release folder. Synthetic speech + screening + windows +
training + evaluation + packaging. Nothing here writes into git.

## External tools (not vendored)

- **spqx** (`~/.local/share/spqx/native-voice-design-*/` locally): Qwen3-TTS
  VoiceDesign for designed voices. The scripts call its binary and read its
  output folders; they do not import it. Apache-2.0 model, voices described in
  text so no real speaker is involved.
- **whisper.cpp server** (`whisper-server -ng -m ggml-small.bin`), used only to
  (a) check that a clip really says the wake word and (b) get word timestamps for
  placing windows. It is a screener, never a training target.
- **Piper** voices whose training corpora are CC0 or CC BY, for a second
  synthetic source with many speakers.
- **uv** for the Python environment (`pyproject.toml`, `uv.lock`).

Data lives outside the repo, by default
`$XDG_DATA_HOME/foxline/waqy-train-<n>/` (speech, features, models) and
`$XDG_DATA_HOME/foxline/waqy-corpus/` (banks). A model store such as
`$XDG_DATA_HOME/byteowlz/models/waqy/` holds finished releases.

## Order of operations

```bash
uv sync                                        # environment
uv run --no-project python -m pytest -q        # screening unit tests

# 1. synthetic speech: Piper (isolated phrases + context sentences)
python build_data.py $ROOT synth hey_oqto 7000 1200
python build_data.py $ROOT asrgate hey_oqto     # keep only clips heard as the wake word

# 2. designed voices from spqx (24 kHz) -> 16 kHz bank, then word timestamps
python prep_shared.py <spqx output dir> $BANK
python realign_bank.py $BANK                    # resumable; per-clip failures isolated
python decide_timings.py $BANK                  # keep/drop with reasons -> clean.jsonl

# 3. windows around the wake word (positives) and around look-alikes (negatives)
WAKE_BANK_OUT=$ROOT/context-shared python context_data.py bank $ROOT $BANK 2 hey_oqto,oqto
python context_data.py owner $ROOT              # owner takes, consented sessions only

# 4. general negatives and held-out validation (MUSAN, VoxPopuli, FLEURS)
python build_data.py $ROOT stream musan
python build_data.py $ROOT stream voxpopuli
python build_data.py $ROOT validation 40
python build_data.py $ROOT negatives 400

# 5. assemble, train, export
python context_data.py assemble $ROOT hey_oqto-gpu1 hey_oqto
python pipeline.py $ROOT hey_oqto-gpu1 config
WAKE_DEVICE=cuda python pipeline.py $ROOT hey_oqto-gpu1 train
python pipeline.py $ROOT hey_oqto-gpu1 export

# 6. evaluate: general false wakes, streaming sentences, owner held-out sitting
python evaluate.py $ROOT hey_oqto-gpu1 --clips hey_oqto --owner $SITTINGS
python sentence_fa.py $BANK hey_oqto-gpu1,hey_oqto-gpu2 --all-voices --dither
python eval_shared.py hey_oqto-gpu1 0.5
python compare_models.py hey_oqto-gpu1,hey_oqto-gpu2 <held-out session prefix>

# 7. package an immutable release
python release.py $ROOT spqx-r3 hey_oqto-gpu1 oqto-gpu5
```

## Publish selected heads to Hugging Face

From the repository root, select exact classifier IDs and a new public release name:

```bash
just publish-heads "$PACK" public-1 hey_hex,hex
# Optional inputs for provenance/consent or an offline catalog merge preview:
just publish-heads "$PACK" public-1 hey_hex,hex \
  --metadata "$REVIEW" --consent "$CONSENT" --catalog "$CATALOG"
```

An optional TOML profile avoids repeating a selection. Keep real profiles in the
private working checkout's `.internal/` directory; it is excluded from GitHub
snapshots. The committed `.internal/head-export.toml` intentionally selects no
heads yet. The public-safe example is `training/head-export.example.toml`:

```toml
[export]
source = "/path/to/immutable/source-release"
name = "public-1"
heads = ["hey_hex", "hex"]
repo = "byteowlz/waqy"
metadata = "head-review.json"
consent = "head-consent.json"
```

```bash
just publish-heads-config .internal/head-export.toml
# Explicit CLI overrides; the same selected-subset/consent checks still apply:
just publish-heads-config .internal/head-export.toml --name public-2 --heads hey_hex
# Only after approving the exact weights and preview:
just publish-heads-config-apply .internal/head-export.toml
```

Equivalently use `publish_hf.py --config PROFILE.toml`. CLI arguments override
profile values. Config paths are relative to the profile's directory; CLI paths
are relative to the calling directory. Profiles are loaded only when explicitly
requested, never auto-discovered. Unknown keys/types are refused; there is no
`apply` or credential setting. JSON schema: `training/head-export.schema.json`.
A selection config is not a publication-consent record and cannot override one.

This defaults to an **offline, read-only dry run**: JSON lists exact destination
paths, hashes, sizes, selected metadata and the proposed catalog. No HF import,
credentials, upload or local curated release is needed. Without `--catalog`, the
catalog preview contains only this release; it is not a claim about remote state.
The shared front end and only the chosen heads are included. Stray ONNX files,
audio, unselected/private heads, internal notes, paths and raw reports are excluded.

Each selected head must explicitly declare boolean `private` and
`owner_recordings`. Unknown provenance is refused, not assumed owner-free. Older
packs may supply a separate reviewed JSON record tied to the source manifest:

```json
{
  "source_manifest_sha256": "<sha256 of PACK/manifest.json>",
  "heads": {
    "hey_hex": {"private": false, "owner_recordings": true},
    "hex": {"private": false, "owner_recordings": true}
  }
}
```

These example flags are not attestations: review the actual training sources.
A review cannot contradict existing flags or override the private roster. Metadata
and consent records must cover **exactly** the chosen head IDs. Unselected runtime
relationships (`implies`/`suppress`) are refused rather than silently changed.

Owner-derived weights additionally need an explicit publication-consent record:

```json
{
  "source_manifest_sha256": "<sha256 of PACK/manifest.json>",
  "repo": "byteowlz/waqy",
  "heads": ["hey_hex", "hex"],
  "publish_derived_weights": false
}
```

Training/enrollment permission is not publication consent. Only the owner may
approve the exact subset and change that final field to `true`; do not infer it
from a broad old consent record. Record bodies and personal IDs stay local: only
digests and the approved scope enter public provenance. Changing source weights,
metadata, destination or selection requires the corresponding scope to be reviewed.
No performance qualification is created by these records; complete the separate
held-out/runtime gates before approving publication. Missing measurements stay missing.

After reviewing the exact plan and receiving weight-publication approval:

```bash
just publish-heads-apply "$PACK" public-1 hey_hex,hex \
  --metadata "$REVIEW" --consent "$CONSENT"
```

The apply recipe uses `huggingface-hub==0.36.0` through uv. The target model repo
must already exist and be public; creating repos or changing visibility is separate.
Apply reads a fresh catalog at a pinned remote commit and sends one exact-file,
parent-fenced Git commit including weights, manifest, catalog, card and legal files.
A concurrent remote change rejects the commit; rerun the preview/check rather than
force-overwrite. Existing release files must match exactly. Prior catalog entries
are preserved; an existing model repository without a catalog needs an audited
backfill first. `--catalog` is offline-only and cannot override that fresh check.

`curate_release.py SOURCE OUT_ROOT NAME --heads ...` remains a local-only export
using the same selection/consent checks (`--dry-run`, `--metadata`, `--consent`).
Neither command retrains, edits source packs, publishes recordings or enables a head
in any host. Run `just check-publication` for offline regression tests.

## Rules that the scripts enforce or assume

- One wake word per clip; drop clips with a repeated word, two speech groups,
  implausible spans or digital silence inside the word (`decide_timings.py`).
- Windows are labelled from word timestamps on the current audio, never from a
  stale alignment.
- A held-out sittings directory is never trained on; `--owner-sessions` makes
  that explicit in the evaluator.
- Synthetic audio is training material; evaluation stays on real recordings and
  on unseen voices.
- `release.py` refuses to overwrite a release folder: a new threshold or head is
  a new release.

## Files

| File | Purpose |
|---|---|
| `build_data.py` | synthetic speech, ASR gate, general negatives, held-out validation |
| `prep_shared.py` | external designed-voice banks → 16 kHz bank |
| `realign_bank.py`, `decide_timings.py`, `check_timings.py`, `fix_timings.py` | word timestamps and screening |
| `context_data.py` | in-context windows, owner takes, assembly |
| `pipeline.py` | config, backgrounds, augment, train, export (CUDA/CPU, warm start) |
| `evaluate.py`, `sentence_fa.py`, `eval_shared.py`, `compare_models.py` | streaming, per-sentence, held-out-voice and owner evaluation |
| `live_review.py` | transcribe a live test session and line up spoken phrases with detections |
| `release.py` | package an immutable release with hashes and provenance |
| `curate_release.py`, `publication.py`, `publish_hf.py` | reviewed per-head subsets, offline publish plans and exact-file HF commits |
| `provenance/` | mel graph generation, TF-checkpoint reader, embedding verification |

## Training your own word (the whole path)

1. **Write the spec** — `words/<name>.json`:
   - `name`, `display` (how it is written in prose) and `spoken_primary` per language (how it is
     **said**; these can differ, see step 3);
   - `address` (variants used for isolated clips) and `address_primary` (used inside sentences):
     the natural way to address the device in that language — de/en "Hey", es "Oye",
     fr "Dis", ko "-ya", ru "Privet" (ADR 0006);
   - `forms` per language, `confusables` per language (what it must *not* fire on),
     `templates` per language (carrier sentences, `{A}` = address, `{W}` = the word);
   - `accept_stems` (optional): every spelling an ASR may produce for the right sound.
2. **Build the inventory** — `python wordpack.py texts words <out.json> --per-word 240`
   (55% addressed phrases, 18% look-alikes; `PRIORITY` at the top orders the batch).
3. **Check the pronunciation before training** — spelling is not sound, and each TTS voice
   renders a spelling differently. Synthesise 3 clips per candidate spelling
   (`synth_spqx.py`), screen them with Whisper for reference, then put them in front of the owner
   on `web/review` (README there). Their ear decides; ASR only advises. Set the winner as
   `spoken_primary`; the losers stay a minority of clips so the head tolerates them.
4. **Synthesise** — `python synth_spqx.py <texts.json> <voice-bank> <out> --workers 3` on a GPU
   host (spqx + a VoiceDesign checkpoint). Fictional designed voices only; no real speaker audio.
5. **Screen and build heads** — `python wordpack_train.py <synth-dir> <out> --words a,b --train`:
   transcribes every clip (accepting `accept_stems`), keeps positives around the word, negatives
   from elsewhere in the clip and from look-alike sentences, augments the training side, then
   trains one small head per address form (`hey_<word>`, `oye_<word>`) plus a bare head.
6. **Pack and evaluate** — `release.py` produces the immutable release; report per word: recall on
   held-out designed voices, false-accepts on that word's confusables, general false wakes/hour.
7. **Bare heads are gated** — they are trained but the runtime only reports them at a speech onset
   after silence, so a bare word cannot trigger inside continuous speech (trx-sg05.3).
