// Export reviewed clips as a labelled manifest for ASR training (e.g. Moonshine), with consent gates.
//
//   REVIEW_DIR=<set> node export.mjs --out manifest.jsonl [--only genuine] [--require-transcript]
//
// Each line: {audio, text, asr_text, label, detector_score, source}. `--only` keeps clips whose
// owner label matches (default: genuine only, because false detections often contain other people,
// TV or music, which are not ours to train on). `--require-transcript` drops clips without a
// corrected transcript, so the export only contains text the owner confirmed.
import fs from 'node:fs';
import path from 'node:path';

const dir = path.resolve(process.env.REVIEW_DIR ?? '.');
const clips = JSON.parse(fs.readFileSync(path.join(dir, 'clips.json'), 'utf8'));
const labels = fs.existsSync(path.join(dir, 'labels.json')) ? JSON.parse(fs.readFileSync(path.join(dir, 'labels.json'), 'utf8')) : {};
const edits = fs.existsSync(path.join(dir, 'transcripts.json')) ? JSON.parse(fs.readFileSync(path.join(dir, 'transcripts.json'), 'utf8')) : {};
const arg = (name, fallback = null) => {
  const i = process.argv.indexOf(name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};
const out = arg('--out', path.join(dir, 'export-manifest.jsonl'));
const only = arg('--only', 'genuine').split(',').map((s) => s.trim());
const requireTranscript = process.argv.includes('--require-transcript');
const canonical = arg('--canonical', 'Oqto');  // adds a normalised form; transcripts stay literal
const toCanonical = (text) => text.replace(/\b(okto|octo|ok-to|octoh?|oqto)\b/gi, canonical);

const rows = [];
const skipped = { unlabelled: 0, otherLabel: 0, noTranscript: 0 };
for (const clip of clips) {
  const label = labels[clip.id]?.label ?? null;
  const text = edits[clip.id]?.text ?? null;
  if (!label) { skipped.unlabelled++; continue; }
  if (!only.includes(label)) { skipped.otherLabel++; continue; }
  if (requireTranscript && !text) { skipped.noTranscript++; continue; }
  rows.push({
    audio: path.join(dir, 'clips', `${clip.id}.wav`),
    text: text ?? clip.heard ?? '',
    asr_text: clip.heard ?? '',
    canonical_text: canonical ? toCanonical(text ?? clip.heard ?? '') : undefined,
    label,
    detector_score: clip.detector_score ?? null,
    source: 'owner-review',
  });
}
fs.writeFileSync(out, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''));
console.log(JSON.stringify({ out, rows: rows.length, only, requireTranscript, canonical, skipped }));
