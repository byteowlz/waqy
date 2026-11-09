# Pronunciation review (listening page)

Spelling is not sound. A wake word can be written one way and said another, and each TTS voice
renders a spelling differently: German "Dschini" came out unvoiced ("Tschini") in some designed
voices, "Djini" gave the voiced affricate, and English "Geni" was heard as "Jenny" while "Genie"
and "Jeanie" were right. Measure and listen; do not guess.

This page is the listening half: it plays labelled clips and records one vote per (language,
round) into `votes.json`, which the training pipeline then reads instead of a chat message.

## Run it

```bash
REVIEW_DIR=~/.local/share/foxline/waqy-review/<set> PORT=8792 REVIEW_HOST=127.0.0.1 bun server.mjs
```

`REVIEW_DIR` contains:

- `clips.json`: `[{id, lang, spelling, text, heard, clip, round?}]` — `heard` is what an ASR
  screener transcribed, shown next to each clip as a reference, never as the decision;
- `clips/<id>.wav`;
- `votes.json` (created on first vote).

`REVIEW_HOST` overrides the optional private `.internal/review-server.toml` host;
without either, the public source binds only to loopback. Hosts are gated
(`REVIEW_ALLOWED_HOSTS`, default `<host>,127.0.0.1,localhost`) and the page is
served without a build step. On the tailnet, register it in owlznest so the owner can reach it.

## The two-round pattern that worked

1. **Round 1 — spellings from the spec.** Synthesise 3 clips per spelling per language (isolated,
   "Hey …", in a sentence) and screen them with an ASR pass for reference.
2. **The owner listens and votes.** Their ear overrules the transcript: a spelling can pass the
   ASR check and still sound wrong.
3. **Round 2 — fix the sound, not the layout.** If a spelling is off, generate candidates that
   target the actual phonemes (for a voiced /dʒ/: Djini, Djinie, Djinni, Dschiini, Djiini, Genie)
   and vote again. Keep round 1's votes; they are the record of why round 2 exists.
4. **Apply the winner** as `spoken_primary` in the word spec: the chosen spelling becomes the
   majority of that word's clips, the alternatives stay a minority so the head tolerates them,
   and `accept_stems` lists every measured spelling so the ASR gate judges sound, not spelling.

Vote keys are `<lang>-r<round>`; never reuse a key across rounds, and never let a Save write an
empty value (it silently erases the previous decision).

## Correcting transcripts and exporting training data

Marking a clip also lets you fix its transcript in place. The ASR text is kept (`clips.json`),
the correction goes to `transcripts.json`, so the original stays auditable and the corrected text
can serve as gold labels.

`export.mjs` writes a manifest for ASR training (e.g. Moonshine):

```bash
REVIEW_DIR=<set> node export.mjs --out manifest.jsonl --only genuine --require-transcript
```

- `--only` filters by owner label; the default is `genuine` **on purpose**: false detections
  often contain other people, TV or music, and those are not ours to train on.
- `--require-transcript` keeps only clips whose text the owner confirmed.
- Each line carries both `text` (corrected) and `asr_text` (original), plus the label and the
  detector score, so a downstream trainer can audit what changed.

## Spelling convention for transcripts

Write what a naive listener would: German **Okto**, English **Octo**. Only write **Oqto** when the
speaker spells the letters out. The brand spelling belongs to the dictionary/normalisation layer,
not to the recogniser; `export.mjs --canonical Oqto` adds a `canonical_text` field that maps
okto/octo/oqto to the product token, while `text` and `asr_text` keep the literal labels.
