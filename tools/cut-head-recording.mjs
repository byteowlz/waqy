#!/usr/bin/env bun
// Cut review candidates only; no automatic negative labels or training ingestion.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { parseArgs } from 'node:util';
import { headReport, headClip } from '../web/app/heads-live.mjs';

const { values } = parseArgs({ options: { session: { type: 'string' }, out: { type: 'string' }, 'marked-only': { type: 'boolean', default: false } } });
if (!values.session) throw new Error('Usage: bun tools/cut-head-recording.mjs --session /private/session-dir [--out /new/private/dir] [--marked-only]');
class ClipError extends Error { constructor(_status, message) { super(message); } }
const directory = path.resolve(values.session), report = headReport(directory);
if (report.meta.mode !== 'heads') throw new Error('Expected a head-explorer recording');
if (report.meta.intent !== 'negative-only' && !values['marked-only']) throw new Error('Mixed sessions require --marked-only');
const marked = new Set(report.marks.filter(m => m.kind === 'false-trigger').map(m => `${m.atSample}:${m.detection}`));
const events = new Map();
for (const chunk of report.chunks) {
  for (const frame of JSON.parse(fs.readFileSync(path.join(directory, chunk.headScores), 'utf8'))) {
    const atSample = (frame.f + 1) * 1280;
    for (const i of frame.d) {
      const key = report.meta.heads[i];
      if (values['marked-only'] && !marked.has(`${atSample}:${key}`)) continue;
      if (!events.has(atSample)) events.set(atSample, []);
      events.get(atSample).push({ key, score: frame.s[i], threshold: frame.t[i], explicitlyMarked: marked.has(`${atSample}:${key}`) });
    }
  }
}
const end = Math.max(0, ...report.chunks.map(c => c.startSample + c.samples));
const output = values.out && path.resolve(values.out);
if (output) fs.mkdirSync(output, { mode: 0o700 }); // Must be new; never overwrite earlier review material.
const manifest = [];
for (const [atSample, heads] of [...events].sort((a, b) => a[0] - b[0])) {
  const startSample = Math.max(0, atSample - 32000), samples = Math.min(end, atSample + 16000) - startSample;
  const row = { sessionId: report.meta.sessionId, atSample, startSample, samples, sampleRate: 16000, heads,
    proposedLabel: 'negative', humanReview: 'pending', trainingEligible: false,
    note: 'Review for actual wake phrases/homophones before using as a negative for each named head.' };
  if (output) {
    const bytes = headClip(directory, startSample, samples, ClipError);
    row.file = `crossing-${String(manifest.length).padStart(6, '0')}.wav`;
    row.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    fs.writeFileSync(path.join(output, row.file), bytes, { mode: 0o600 });
  }
  manifest.push(row);
}
if (output) fs.writeFileSync(path.join(output, 'manifest.jsonl'), manifest.map(row => JSON.stringify(row)).join('\n') + (manifest.length ? '\n' : ''), { mode: 0o600 });
console.log(JSON.stringify({ sessionId: report.meta.sessionId, crossings: manifest.length,
  headAssociations: manifest.reduce((n, row) => n + row.heads.length, 0), dryRun: !output, output,
  trainingEligible: false, registration: 'Register any retained training/evaluation subset in the central dataset registry before use.' }));
