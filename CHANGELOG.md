## Unreleased

### Private working source and optional head-export profiles

- Adds a reviewed public-source snapshot policy while keeping working history,
  deployment defaults, agent/context notes and `.internal/` export profiles private.
  Local deployment recipes and review-server bind preferences retain their existing
  private defaults; the public review server defaults to loopback.
- Adds explicit `--config` TOML profiles and `publish-heads-config` preview/apply
  recipes. CLI values override config; config-relative paths, strict keys/types
  and a public JSON schema make profiles reusable without enabling publication.
  No heads are selected automatically and config cannot grant weight consent.

### Publish explicit head subsets

- Adds `just publish-heads SOURCE NAME HEADS` for an offline JSON preview and
  `just publish-heads-apply` for an explicitly approved Hugging Face publication.
  Only selected classifiers, the shared front end, manifest, catalog and legal/card
  files enter the exact upload list; stray ONNX files and recordings are excluded.
- Refuses private heads, unknown owner provenance, hash/path errors and changed
  immutable releases. Owner-derived weights require exact source/subset/repository
  consent, not a broad training permission. Public metadata is allowlisted.
- Publishes the selected release and append-only catalog in one parent-fenced commit.
  Reuses the same checks for local curation; no heads are trained, uploaded or enabled
  by installing this tooling. See `training/README.md`.

## 2026-11-05

### Yo greeting across the core assistant words

- Extends Yo alongside Hey to Oqto, Pi, Elissa, OpenCode, Jarvis, Computer and Home Assistant,
  in addition to Hex. German/English collection forms and synthesis variants stay in sync;
  command phrases are unchanged. All eight are covered by inventory regression tests.

### Next synthesis inventory: Home Assistant and Yo Hex

- Adds German/English addressed-only Home Assistant with explicit spoken spelling and accepted
  orthographic variants. Adds Yo Hex as an alternative greeting pooled into the existing
  addressed Hex head, preserving Hey Hex; no separate word/head is created.
- Address variants now reach synthesis instead of being treated as bare spellings or ignored.
  Collection, owner weighting, trainer, phoneme audit and gates recognise Yo consistently.
  Existing Okay/Ok collection phrases are also correctly tagged as addressed rather than bare.
  These are collection/synthesis specifications, not newly trained or published detectors.

### Continuous head-explorer false-trigger recording

- `/heads` can opt into one continuous negative-only or mixed test session with all selected
  compatible heads. Bounded audio buffers stream to the private host; no audio saves by default.
- Logs retain sample-exact chunk offsets, every head score and threshold per frame, rising
  crossings, release-qualified IDs and classifier hashes. Filter/pagination never limit logging.
  Stop drains pending inference; recording fences Reset. Save failures stop capture and retain
  failed chunks for retry rather than dropping audio silently.
- Details can replay a crossing across chunk boundaries, mark it as false on that exact head,
  and download the session index. `tools/cut-head-recording.mjs` cuts per-head review candidates
  from long sessions; no automatic negative labels, training consent or publication.

### Correct acoustic stress evaluation

- `gate_rooms.py` reports bare and addressed heads separately, with wake and false-fire
  numerators/denominators. Its first report mistakenly preferred addressed heads for bare takes;
  those earlier aggregate figures must not be used as robustness evidence.
- Adds -12/-24 dB attenuation without renormalisation, plus reverb/noise conditions, frozen
  other-word negatives and incremental checkpoints. Simulated RIRs can overlap the training bank:
  this is stress testing, not unseen-room or physical far-field qualification.
- `just check-room-gate` covers head eligibility, false fires, missing evidence, audio format,
  attenuation and transforms using synthetic fixtures only.

### Live test picker, and store caps that survive five microphones

- `/detect` offers a release picker (installed release or any published pack) and a checkbox per
  head; only the ticked heads are loaded, so several wake words can be compared live in one pass.
  Heads come from `/packs/<release>/`, hash-verified by the same path the head explorer uses.
- The recording store refused saves at exactly 5000 takes ("Recording store is full"), which a
  five-microphone session reaches quickly (owner: 5000 takes, 206 MB of a 1 GiB byte budget).
  Defaults are now 50k takes / 8 GiB, both settable per host (`RECORDINGS_MAX_COUNT`,
  `RECORDINGS_MAX_BYTES`), a non-numeric value falls back to the defaults, and the summary exposes
  `capacity` so the takes header warns at 90% instead of failing at the next save.

## 2026-11-04

### Record the same phrase on several microphones at once

- Options -> "Record on" lists the input devices; several can be ticked. Each phrase then saves
  one take per device, tagged with the device label (`mic` in the take metadata, counted per mic
  in `/api/recordings/summary` as `byMic`). One sitting yields several acoustics of the same
  words instead of several sittings.
- Per-device timing: each take is cut to its own speech span (onset with a 150 ms margin), so an
  input that starts late (Bluetooth can lag 100-200 ms) still aligns on the phrase.
- A device that cannot be opened is dropped after the block with a message; a device that hears
  nothing is named in the status line. The microphone list is remembered with the other settings.
- Desktop browsers only: iOS Safari and most Android browsers allow a single input at a time.
- Checks: `multi-mic-check.mjs` records on two independent synthetic streams (one started 150 ms
  late) plus a broken device: 8 takes, both labels, aligned within 200 ms, per-mic counts, broken
  device dropped, zero page errors. The single-mic browser round trip still passes (24 checks).
  Physical hardware and Bluetooth were not tested.

## 2026-11-03

### Look-alikes that contained the wake word

- 34 spec look-alikes were the wake word itself: article fragments ("Elisa, die",
  "Colonel, der", 13 of them), possessives ("Miller's", "Hal's"), "miller lite", "by the way in",
  "mei ling", accepted spellings (genie/jeannie/jeanie for Ginee) and plain repeats ("Codec").
  As hard negatives they teach a detector to ignore its own word. Removed from the specs.
- `action.json` listed "Aktion" as wake form, accepted spelling and look-alike at once; its
  pronunciation note says it is a distinct look-alike, so it is now only that.
- One rule in `build_collection.py` (fails the build) and in the editor (refuses the add): a
  look-alike may not contain the language's wake form as whole words, nor equal an accepted
  spelling. accept_stems are language-agnostic ASR stems, so address words (hey, okay) in them
  do not count.
- Editor rows are one grid: every action button is the same size (120x40, 104x40 on phones),
  main actions share one column, take counts sit under the phrase instead of wrapping buttons.

### Look-alike editor (`/lookalikes`)

- One page lists all look-alikes (304 across 38 words) per language, each beside the word's
  wake forms. Remove, restore and add with Undo; Back to recording returns to the recorder,
  which now loads the edited list.
- Same sound is never a look-alike: adding a wake form or an accepted spelling is refused.
  `build_collection.py` now carries each spec's `accept_stems` into `wordpack.json`.
- `wordpack.json` was stale: it still listed homophones the specs dropped in a3d5bb2
  (Jeff, Kit, kit kat, buy the way, byeway, side quest). Regenerated; the owner had
  recorded 3 "buy the way" takes as negatives, which the editor lists for deletion.
- Edits are a private overrides file with revision checks (a stale tab gets a conflict, not
  a silent overwrite), same-origin writes, and an export for the next training run.
  Deleting takes is offered only for removed or unlisted phrases, in two steps.
- Robustness: summary, listing and look-alike views no longer return 500 when the
  recordings folder is removed under a running server; the overrides folder is recreated.
- Tests: 41 unit/server tests and an 8-check Chromium round trip (synthetic seeded take,
  isolated stores). Text contrast measured at 7.66:1 or better.

## 2026-11-02

### All-word recorder: no stuck quiet takes

- Nonzero low-level takes now save with a review hint instead of being discarded
  and retried at the same queue position. Peak level alone does not measure SNR
  or establish that speech is unusable.
- Missing or digitally silent input pauses with an explicit reason; the current
  take stays missing. Save failures also pause instead of endlessly re-recording,
  and uploads have an eight-second deadline. Resume explicitly retries.
- Regression check covers All words with three takes per phrase, attenuated audio
  starting on take three, failed uploads, and a disconnected/silent source. Uses
  synthetic fake-microphone audio; no physical-device verification is claimed.

## 2026-10-30

### Listen back and prune collected takes

- The collection page lists the saved takes for the selected word with phrase, language,
  wake/look-alike, form, style and duration. Each row plays its private WAV and can be
  deleted after a confirmation; deleting removes the WAV and metadata and refreshes coverage.
- Read-only routes: `GET /api/recordings/takes/<word>` (path, not a query, since the server
  refuses query routes) and `GET /api/recordings/<id>/audio` (`audio/wav`, `no-store`, id
  validated). Coverage now comes from a whole-store summary, so the capped listing can no
  longer hide earlier takes.
- Tests: 21 server/store tests plus 13 Chromium checks (fake microphone) covering playback,
  delete-and-file-removal, filters and the same-origin boundary. Deletion stays local.

## 2026-10-30

### Owner collection: every wake word plus look-alikes

- The recording page is now word- and spec-driven: 26 wake words, each with its
  natural addressed/bare forms and its confusables per language, generated from
  `training/words/*.json` by `training/build_collection.py` into
  `web/app/wordpack.json`. Hand-copied phrase lists are gone.
- Takes carry `word` and `form` (addressed/bare/confusable) so training and the
  coverage grid can group them. New `/api/recordings` read listing and a coverage
  panel show how many takes exist per word and session.
- Why: the deployed heads were trained only on designed synthetic voices, so the
  owner's voice is out of distribution (hex missed, elissa/pi/otacon unreliable),
  and `oqto` never saw other wake words as negatives (opencode/otacon also fired
  it). This collector gathers the owner's positives and hard negatives per word.
- Verification: 19 server/store tests plus a 10-check Chromium run (fake microphone)
  covering word selection, addressed/bare/confusable plans, saved metadata,
  coverage updates, English plan differences, same-origin uploads and zero errors.
  Saving remains private, unreviewed and training-ineligible.

## 2026-10-30

### Viewport head explorer

- `/heads` now uses the viewport for labelled threshold lanes, with responsive
  pagination instead of page scrolling. Real wake phrases, release revisions and
  per-head file sizes remain visible; release/details panels are secondary.
- Added filesystem-measured model bytes, rolling mean/p95 processing time,
  realtime-budget estimate, audio backlog and input level. File bytes are not RAM;
  processing wall time divided by audio time is not OS CPU utilization.
- Preserved release choices and page-local threshold editing. Start/Stop/Reset
  fence late microphone grants and stale frames; peaks include every score frame,
  crossings use bounded rising-edge history, and excessive backlog stops the test.
- 19 unit/server tests and 27 browser checks pass across desktop, phone, landscape
  and small/zoom-equivalent viewports. Actual WASM checks use a synthetic fake mic;
  physical devices and OS CPU measurement are not claimed. No audio is uploaded.

## 2026-10-29

### Head explorer (`/heads`)

- New page on the dev UI: every release under `WAQY_PACKS_DIR` is offered, and every head of every
  selected release is scored in one browser pass, so packs can be compared directly. Shows live
  score, session peak, an editable per-head threshold and a crossing log; records nothing and
  uploads nothing.
- `packsAssets()` serves `/api/releases` and `/packs/<release>/<file>`, and offers a release only if
  every classifier, mel and embedding hash in its manifest matches. Directory names must start with
  an alphanumeric character, so hidden or traversal-shaped names are ignored.
- `heads-check.mjs` proves it in headless Chromium with a fake microphone: 24 heads load from one
  release, the "hey snake" fixture peeks at 0.95 (bare) and 0.91 (addressed), 4.83 ms per 80 ms
  frame for all 24 heads, no page errors and no uploads.
- Registered as the private head-explorer service; deployment addresses stay in private configuration.

## 2026-10-29

### Word pack spqx-pack1 (24 heads)

- 12 words with natural address and bare form, German and English: elissa, pi, opencode, hex,
  governor, colonel, snake, mei-ling, meryl, otacon, miller, naomi.
- Every head verified on its own held-out designed voices: median 0.86-0.96 above threshold.
- Cross-word confusion: 0 of 132 head/pair cells above 10%, worst cell 0.07 (was 66 cells, worst
  0.98 before ADR 0007). `waqy check` passes on 24 heads, 6.1 MB total.
- Runtime smoke test through the Rust detector at 16 kHz: "hey snake" clip scores 0.88 addressed and
  0.95 bare. Borderline cross-head scores exist on single clips (a "hey pi" clip also puts elissa at
  0.51), which is what per-head thresholds (ADR 0005) are for.
- Owner recordings are not part of this pack.

### Word-pack training

- Heads derive their config from the head name (`hey_elissa` used to raise KeyError in `pipeline.py`).
- The default recipe collapses smaller word-pack sets to constant-negative: `max_negative_weight`
  ramping to 3000 with 1024 general negatives per batch produced 24 heads that predicted ≤0.002 on
  their own training positives. A probe with `max_negative_weight: 1` and balanced batches
  (200/200/200/50) learns the same features cleanly, so the runner now uses that recipe and says why.
- ADR 0007: addressed heads train with cross-word adversarial negatives. Confusion between wake
  words was 66 of 132 cells above 10% (worst 0.97-0.98); with the other words in the negative set it
  is 0.02-0.06 at unchanged recall.
- Fictional child voices (`training/children_voices.py`) as hard negatives: the deployed head is
  triggered by children because every training voice is an adult, and children's speech is out of
  distribution. Designed voices keep minors' recordings out of the data.

# Changelog

All notable changes to waqy are documented here.

## [Unreleased]

### Added

- **word packs**: per-word specs (`training/words/*.json`) with the language's natural address
  form (de/en "Hey", es "Oye"; fr "Dis", ko "-ya", ru "Privet" recorded for later), confusables
  and carrier sentences; `wordpack.py texts`, `synth_spqx.py` (designed voices, cross-lingual
  reference available for accent experiments) and `wordpack_train.py` (screening, per-head
  windows, training). ADR 0006.

- **detector**: manifest-driven `EmbeddingHeads` backend (frozen log-mel + speech-embedding front end, one classifier per phrase) behind a `WakeDetector` interface, with the `wake-stream/1` process protocol (`waqy listen`) and `waqy detect-file` for offline runs.
- **browser**: `web/waqy-engine.mjs`, the same state machine in onnxruntime-web/WASM, with parity tests and a half-duplex playback gate (the host's own TTS otherwise wakes the detector at 0.98).
- **training**: `training/` pipeline — synthetic speech from spqx designed voices and Piper, screening (one wake word per clip, plausible word span, no silence inside the word), in-context windows labelled from word timestamps, owner takes under explicit consent, training on CUDA or CPU, streaming and per-sentence evaluation, and `release.py` packaging.
- **contract**: inaudible input dither (ADR 0004) in both runtimes; near-miss reporting (`--report-above`) for diagnostics only.
- **docs**: ADRs 0001–0005 (evidence-only detections, process protocol, models-as-data, dither, per-head thresholds) and `CONTEXT.md`.

### Fixed

- Mixed float16/float32 feature batches reached the convolution as half precision and aborted training; batches are cast to float32 on load.
- Streaming look-alike sentences after digital silence produced false wakes that per-window evaluation had hidden.
- Isolated synthetic clips carried a stale energy span (one 5.8 s clip reported a word end at 8.54 s); timings are now derived from word timestamps on the current files, with screens.
