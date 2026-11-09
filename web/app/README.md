# waqy browser UI (standalone)

Three private pages served by one small Bun server, for testing releases and for
collecting the owner's own recordings. Nothing here is required to use waqy; it is
the reference host and the test bench the models were developed against.

- `/` **record phrases**: one word or all words, wake phrases and/or look-alikes.
  Queues only missing takes per phrase/style from the server summary. Stop and
  resume across sittings; Fast and Relaxed pacing, pause/skip/redo controls.
  Writes one WAV plus metadata per take (normal, whispered, from across the room).
  Low peak level is a review hint, not a rejection gate: nonzero quiet takes save
  and advance. Missing/digitally silent input pauses with an explicit reason.
  Failed/timed-out saves pause rather than automatically re-recording forever;
  Resume explicitly retries. Upload requests have an eight-second deadline.
- `/lookalikes` **review look-alikes**: every word's near-miss phrases per language, shown
  next to its wake forms. Remove, restore, add; Undo after each edit. Adding the wake word
  or one of its accepted spellings (`accept_stems`) is refused: same sound is a wake form,
  never a look-alike. Recorded phrases that are removed or no longer listed show their take
  count and can have those takes deleted in two steps. Edits live in a private overrides
  file (`LOOKALIKES_FILE`) on top of `wordpack.json`; `/wordpack.json` serves the edited
  list, so the recorder queues the new list immediately. Existing heads do not change.
- **Multi-mic**: under Options, "Record on" lists every input device; tick several and the same
  phrase is captured on all of them in one sitting (one take per device, tagged with the device
  label). Each device's take is cut to its own speech span, so a later-starting input
  (Bluetooth, a USB interface) still lines up on the phrase. A device that cannot be opened is
  dropped with a message and the session continues; a device that hears nothing is reported per
  take. Desktop browsers only: iOS Safari and most Android browsers allow one input at a time.
- `/detect` **test the detector**: live detection in the browser. Pick a release (the installed
  one or any published pack) and tick exactly the heads you want to test: the page loads that
  subset and nothing else, so several wake words can be compared side by side in one pass. The
  submode (onnxruntime-web,
  WASM, single thread) with per-head scores, a half-duplex gate for the host's own
  playback, and three save modes: nothing, only a few seconds around each detection
  (all-day), or everything for later transcription.
- `/heads` **compare heads**: viewport-fitted score/threshold lanes, per-head model
  bytes, peak markers and compact runtime statistics. Small/zoomed viewports use
  pagination instead of hiding rows. Release selection and threshold edits stay
  page-local. **Record audio** opts into continuous private recording (up to two hours).
  Default session type: **False-trigger hunt**, intended negatives. All selected heads keep
  processing regardless of filters/pages. Stop drains inference and saves the final partial chunk.
  Details offers crossing replay, per-head false-trigger marks and a session-index download.
  Audio is unsaved by default. Playback is disabled while capturing; Reset is disabled while recording.
  Save failures stop capture visibly and retain failed chunks for **Retry saving**; keep the tab open.
- `/api/live/*` and `/api/recordings/*` store sessions privately on the host.

## Cutting false-trigger candidates

Record one negative-only session with every compatible release selected. Audio is stored as
contiguous 16 kHz PCM16 WAV chunks under `LIVE_DIR/<session-id>/`, with sample offsets in
`chunks.jsonl`. Each chunk has a `.scores.json` sidecar: absolute 80 ms frame index `f`,
full-precision scores `s`, per-frame thresholds `t`, and rising-crossing column indices `d`.
`meta.json` maps columns to `release::head` keys and classifier SHA-256 hashes; marks retain the
same key and exact sample position. Unscored tails are explicit via `processedSamples`.
No four-head truncation is applied to these logs. The UI keeps only 500 recent crossings;
all crossings remain on disk. Keep the tab open and use Stop rather than closing mid-recording.

```bash
# Dry run: list the number of proposed clips, without writing audio.
bun tools/cut-head-recording.mjs --session /private/live/<session-id>
# New private output directory only; group heads crossing at the same sample into one clip.
bun tools/cut-head-recording.mjs --session /private/live/<session-id> --out /private/review/new-cuts
# Mixed sessions require --marked-only, which selects explicitly marked head/event pairs.
```

Clips retain two seconds before and one after each crossing, including across chunk boundaries.
Missing audio is an error, never silently padded. Outputs are **review candidates**, not automatic
training negatives: actual wake words and homophones must not become negatives. The manifest
keeps `trainingEligible: false`; register any retained training/evaluation subset in the central
dataset registry before use. Nothing goes to public hosting or model publication.

## Run

```bash
cd web/app
WAQY_MODELS_DIR=~/.local/share/byteowlz/models/waqy/spqx-r3 \
WAQY_PUBLIC_ORIGIN=https://waqy.example.net \
WAQY_ALLOWED_HOSTS=waqy.example.net \
bun server.mjs
```

| Variable | Meaning |
|---|---|
| `WAQY_MODELS_DIR` | release folder (manifest + files); served at `/models/`, hashes verified on start |
| `WAQY_PACKS_DIR` | parent of immutable releases for `/heads`; `/api/releases` includes verified file byte counts |
| `WAQY_WEB_DIR` | the waqy web engine (`../` by default, so `waqy-engine.mjs` and onnxruntime-web resolve) |
| `WAQY_PUBLIC_ORIGIN` | exact origin the pages are served from; mutations from any other origin are refused |
| `WAQY_ALLOWED_HOSTS` | host header allowlist (default `localhost,127.0.0.1`) |
| `PORT` | listen port, bound to `127.0.0.1` (put TLS in front) |
| `RECORDINGS_DIR`, `LIVE_DIR` | private stores; default under `$XDG_DATA_HOME/foxline/` |
| `LOOKALIKES_FILE` | owner look-alike overrides (JSON, mode 0600); default `~/.local/share/foxline/waqy-lookalikes-1/overrides.json` |

Serve it behind TLS yourself (a reverse proxy with a certificate for
`WAQY_PUBLIC_ORIGIN`); the server sets a strict CSP, cross-origin isolation and
`no-store` for pages and APIs.

## Our dev deployment

On this machine the live instance runs from this checkout in a tmux session, with the
private stores outside the repo and TLS at the reverse proxy:

```bash
just deploy-dev     # restart in tmux 'waqy-dev', then check health and the served release
just dev-log        # attach to the session
just dev-stop       # stop it (TLS and stores untouched)
```

Defaults: port 8036, the configured dev hostname, release
`$HOME/.local/share/foxline/waqy-models/spqx-r3`, stores under
`$HOME/.local/share/foxline/`. Override with `WAQY_DEV_PORT`, `WAQY_DEV_HOST`,
`WAQY_DEV_MODELS`, `WAQY_DEV_SESSION`.

## What it stores, and what it does not

- Recordings: `$RECORDINGS_DIR/<id>/{audio.wav,meta.json}`, mode 0700, 4 s per take,
  silent takes refused. Saving is **not** consent for training, TTS or enrollment;
  `trainingEligible` stays false and review stays pending.
- Live tests: audio chunks plus the detector's scores (`scores.jsonl`) or only short
  snippets around detections and "missed it" presses, plus a listening-time record.
- Never: model files in git, transcripts of private audio, or anything uploaded
  elsewhere.

## Head explorer statistics and limits

- Model size is artifact bytes: one compatible shared frontend plus every selected
  classifier instance. It is not RAM or total browser download size. The API reports
  `head_bytes` and `shared_bytes` from the verified files, not model-card estimates.
- Mean/p95 processing time uses a bounded rolling window of up to 120 audio frames,
  including warm-up. Realtime-budget usage is worker wall time divided by audio
  duration; it is an **estimate**, not device-wide CPU utilization. OS CPU/RAM are
  not measured by this browser UI.
- Input level is the latest 80 ms capture frame peak. Audio backlog is sent minus
  processed audio; over three seconds stops the test instead of accumulating PCM.
- Every score frame contributes to peaks; recent threshold crossings are bounded
  to eight rising edges. These are not the detector's calibrated wake events.
- New runs/reset clear measurements, retain page-local thresholds, and fence stale
  worker messages. Stop also fences a microphone grant arriving after cancellation.
- All existing release choices remain available. Different frontend hashes cannot
  share one worker pipeline and must be compared separately.
- The shared user-directed visual record is in Design Studio under
  `artifacts/waqy/head-explorer-viewport-r1/`; this is Web evidence, not native UI.

## Checks

```bash
# From the repository root, after `cd web && bun install --frozen-lockfile`:
just check-ui                              # build pinned resampler, then routes/stores/guards
just check-web                             # all browser/server unit tests
PLAYWRIGHT=<path to playwright-core> CHROMIUM=<chromium binary> \
  TEST_ORIGIN=http://127.0.0.1:8037 RECORDINGS_DIR=<tmp> SERVER_LOG=<tmp>/server.log \
  bun browser-check.mjs                    # recording page end to end (needs AUDIO_FIXTURE)
PLAYWRIGHT=… CHROMIUM=… AUDIO_FIXTURE=<wav> TEST_ORIGIN=http://127.0.0.1:8037 \
  RECORDINGS_DIR=<disposable-store> bun recording-regression.mjs # third-take faults, never live store
PLAYWRIGHT=… CHROMIUM=… TEST_ORIGIN=http://127.0.0.1:8048 RECORDINGS_DIR=<tmp> \
  LOOKALIKES_FILE=<tmp>/overrides.json bun lookalikes-check.mjs   # editor round trip
PLAYWRIGHT=… CHROMIUM=… LIVE_DIR=<tmp> TEST_ORIGIN=… POSITIVE_WAV=<clip> bun detect-check.mjs
PLAYWRIGHT=… CHROMIUM=… TEST_ORIGIN=… POSITIVE_WAV=<synthetic-hey-snake.wav> \
  EVIDENCE_DIR=<private-evidence-dir> bun heads-check.mjs
```

`speech.json` is the phrase protocol for the recording page (what is displayed, what
is spoken, which takes are positives). Replace it for your own vocabulary; the
detector page reads its phrases from the release manifest instead.
