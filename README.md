# waqy

Wake-phrase detection as data. A frozen speech-embedding front end plus one tiny
classifier per phrase, packaged as an immutable **release folder** that any host
loads: a Rust process, a Rust library, or the browser over WASM. Swapping models
is a path change, never a code change.

A detection is evidence that a phrase was probably spoken. It is never
authentication, speaker identity or permission to act.

Licence: Apache-2.0 (see `LICENSE` and `NOTICE` for third-party assets).

## What it is made of

| Part | Size | Notes |
|---|---|---|
| log-mel front end | 1.1 MB | generated from formulas, no learned weights (`training/provenance/build_mel_onnx.py`), matches openWakeWord's v0.5.1 mel within 1.5e-5 dB |
| speech embedding | 1.3 MB | Google `speech_embedding` (Apache-2.0), verified layer by layer against the TF Hub release |
| one head per phrase | 160 KB (small) / 950 KB (medium) | trained only on the frozen front end's features |
| CPU | ~3.5 ms per 80 ms of audio for two heads | single-threaded WASM on an M2; ~0.18 ms per extra head |

## Use a release

```bash
waqy check --manifest ~/.local/share/byteowlz/models/waqy/spqx-r3   # hashes, protocol, phrases, thresholds
waqy bench --manifest …                                             # CPU per 80 ms frame
waqy detect-file --manifest … clip.wav                              # offline, one JSON line per detection
waqy listen --manifest …                                            # host protocol on stdin/stdout
```

A host (eaRS, Oqto) either spawns `waqy listen` and speaks `wake-stream/1`, or
links the crate and calls the `WakeDetector` trait.

## The protocol (`wake-stream/1`)

stdin, binary frames: `kind: u8`, `len: u32 LE`, payload.

| kind | payload | meaning |
|---|---|---|
| `A` | s16le mono 16 kHz PCM | audio, any length |
| `P` | 1 byte | playback gate: 1 while the host speaks, 0 after |
| `R` | none | forget audio context |
| `Q` | none | quit |

stdout, JSON lines:

```text
{"type":"ready","protocol":"wake-stream/1","sample_rate":16000,"phrases":["hey_oqto"],"detector":"embedding-heads","release":"spqx-r3"}
{"type":"detection","phrase":"hey_oqto","score":0.86,"at_sample":123456}
{"type":"candidate","phrase":"oqto","score":0.31,"at_sample":123456}   # near miss, only with --report-above
{"type":"error","message":"…"}
```

stderr is free-form logging. The wire id stays `wake-stream/1` across renames of
this tool.

## Bring your own model

1. Train a classifier over the frozen front end: input `(batch, 16, 96)` float32
   embeddings, output one score. `training/` does this end to end.
2. Export to ONNX and write a `manifest.json` (file names, sha256 per file,
   threshold per head, provenance).
3. `waqy check --manifest <dir>`; then point your host at the folder.
4. The identical folder loads in the browser engine, so one artifact serves
   desktop and web.

## Browser UI (standalone)

`web/app/` is a small Bun server with two pages: `/` records the owner's phrases
(with a speaking style) into a private store, and `/detect` scores a release live in
the browser, with per-head scores, a half-duplex gate and three save modes. It is
the reference host and the bench the models were developed on; nothing else depends
on it. See `web/app/README.md` for the environment variables and the checks.

```bash
just ui          # http://127.0.0.1:8080, uses $WAQY_MODELS/spqx-r3
```

## Training your own word

`training/README.md` has the full path: write a word spec, build the inventory, **check the
pronunciation with the owner before training** (spelling and sound differ, and each TTS voice
renders a spelling differently), synthesise with designed voices, screen, train one head per
address form, then pack and evaluate. The listening tool for the pronunciation decision ships as
`web/review`.

`training/` holds the whole pipeline: synthetic speech, screening, in-context
windows, owner recordings under explicit consent, training, evaluation and
release packaging. Read `training/README.md`; it names the external tools it
expects (notably **spqx** for designed voices — see below) and where data lives.

Synthetic positives are generated with [spqx](https://github.com/byteowlz/spqx)
running Qwen3-TTS VoiceDesign (Apache-2.0): voices are described in text, so no
reference recording of a real person is involved. Piper voices whose training
data are CC0/CC BY are used as a second synthetic source. Release data and
models never live in this repository.

## Source and model publication

GitHub carries reviewed source snapshots, not the private working history or
operational configuration. Real head-export profiles and deployment defaults live
under `.internal/` in the private checkout and are excluded from public snapshots.
A sanitized TOML example and JSON schema ship with the publishing tools.

Selecting source for GitHub is separate from selecting model heads for Hugging
Face. Head exports default to an offline preview; uploading weights requires an
explicit operation and any applicable exact-subset consent. See the
[training guide](training/README.md#publish-selected-heads-to-hugging-face).

## Status and limits

- Two heads are in production use inside eaRS dictation ("hey oqto" starts
  dictation, "oqto stopp" pauses it) and on a private browser test page.
- Measured on held-out channels: 0.94 recall on 64 unseen designed voices for the
  current "hey" head, 18/18 owner takes, 0/36 owner look-alikes; ~0.12 false
  wakes per hour of unrelated audio.
- German and English are covered; a Spanish wake set exists untrained.
- Per-head thresholds and head relationships are specified but not yet
  implemented in the loaders.
- The browser engine and the Rust backend agree with the Python training path
  within 0.0011 / 0.0004 on identical audio; a third-party reimplementation of
  the front end would need its own parity check.
