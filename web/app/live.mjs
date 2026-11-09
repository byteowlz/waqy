import fs from 'node:fs';
import path from 'node:path';
import { headMetadata, headReport, headClip, saveHeadChunk, checkHeadSession } from './heads-live.mjs';

// Private store for recorded live detector tests: continuous audio in 10 s chunks plus
// the detector's per-80 ms scores and the owner's marks ("missed it", "false trigger").
// Used afterwards to transcribe (Whisper/Parakeet) and line up spoken wake words with
// detections. Evaluation material only: trainingEligible false, never in git.
export const DEFAULT_LIVE_ROOT = path.join(process.env.HOME ?? '/', '.local/share/foxline/wake-live-1');
export const CHUNK_SECONDS = 10;
const MAX_CHUNK_SAMPLES = 16000 * (CHUNK_SECONDS + 1);
const MAX_SESSION_CHUNKS = 6 * 60 * 2; // 2 h per session
const SESSION = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const MARKS = ['missed', 'false-trigger', 'correct'];
const SNIPPET_KINDS = ['detection', 'missed'];
const MAX_SNIPPET_SAMPLES = 16000 * 8;
const clean = (value, max) => typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]/g, '').slice(0, max) : '';

export class LiveError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

// PCM16 mono 16 kHz WAV only; silence is allowed here (a quiet room is real data).
export function parseLiveWav(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = offset => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  const ok = bytes.byteLength >= 46 && tag(0) === 'RIFF' && view.getUint32(4, true) === bytes.byteLength - 8 && tag(8) === 'WAVE'
    && tag(12) === 'fmt ' && view.getUint32(16, true) === 16 && view.getUint16(20, true) === 1 && view.getUint16(22, true) === 1
    && view.getUint32(24, true) === 16000 && view.getUint16(34, true) === 16 && tag(36) === 'data'
    && view.getUint32(40, true) === bytes.byteLength - 44;
  if (!ok) throw new LiveError(415, 'Expected PCM16 mono 16 kHz WAV');
  const samples = (bytes.byteLength - 44) / 2;
  if (samples > MAX_CHUNK_SAMPLES) throw new LiveError(413, `Chunk longer than ${CHUNK_SECONDS + 1} s`);
  return samples;
}

export function createLiveStore({ root = DEFAULT_LIVE_ROOT, maxBytes = 8 * 1024 ** 3 } = {}) {
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  fs.chmodSync(root, 0o700);
  const sessionDir = (id, create = false) => {
    if (!SESSION.test(id ?? '')) throw new LiveError(400, 'sessionId must be a page-session UUID');
    const dir = path.join(root, id);
    if (!fs.existsSync(dir)) {
      if (!create) throw new LiveError(404, 'Unknown session');
      fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    }
    return dir;
  };
  const usage = () => {
    let bytes = 0;
    if (!fs.existsSync(root)) fs.mkdirSync(root, { recursive: true, mode: 0o700 });
    for (const entry of fs.readdirSync(root)) {
      if (!SESSION.test(entry)) continue;
      for (const file of fs.readdirSync(path.join(root, entry))) bytes += fs.statSync(path.join(root, entry, file)).size;
    }
    return bytes;
  };
  return {
    root,
    // meta: { sessionId, seq, startSample, release, threshold, scores: [{f, s:{name:score}, d:[names]}] }
    chunk(bytes, meta, { userAgent } = {}) {
      const samples = parseLiveWav(bytes);
      const seq = meta.seq, startSample = meta.startSample;
      const head = headMetadata(meta, samples, LiveError);
      if (!Number.isInteger(seq) || seq < 0 || seq >= MAX_SESSION_CHUNKS) throw new LiveError(400, 'seq out of range');
      if (!Number.isInteger(startSample) || startSample < 0) throw new LiveError(400, 'startSample must be a sample index');
      if (!Array.isArray(meta.scores) || meta.scores.length > 400) throw new LiveError(400, 'scores must be a short array');
      if (usage() + bytes.byteLength > maxBytes) throw new LiveError(507, 'Live store is full');
      const dir = sessionDir(meta.sessionId, true);
      const metaPath = path.join(dir, 'meta.json');
      checkHeadSession(dir, head, LiveError);
      if (!fs.existsSync(metaPath)) {
        fs.writeFileSync(metaPath, JSON.stringify({
          ...head, captureStartedAt: clean(meta.startedAt, 40),
          sessionId: meta.sessionId, startedAt: new Date().toISOString(), inputKind: 'microphone', role: 'owner-live-detector-test',
          release: clean(meta.release, 80), threshold: head ? null : Number(meta.threshold), userAgent: clean(userAgent, 300),
          chunkSeconds: head?.chunkSeconds ?? CHUNK_SECONDS, humanReview: 'pending', trainingEligible: false,
          note: 'Recorded live detector test. Evaluation only (transcribe, find missed and false detections); not training, enrollment or upload consent.',
        }, null, 2), { mode: 0o600 });
      }
      if (head) return saveHeadChunk(dir, bytes, meta, samples, LiveError);
      const name = `chunk-${String(seq).padStart(6, '0')}.wav`;
      const target = path.join(dir, name);
      if (fs.existsSync(target)) throw new LiveError(409, 'Chunk already saved');
      fs.writeFileSync(target, bytes, { mode: 0o600 });
      const rows = meta.scores.map(row => ({ f: Number(row.f), s: Object.fromEntries(Object.entries(row.s ?? {}).slice(0, 4).map(([k, v]) => [clean(k, 20), Number(v)])),
        d: (row.d ?? []).slice(0, 4).map(x => clean(x, 20)) }));
      fs.appendFileSync(path.join(dir, 'scores.jsonl'), rows.map(r => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''), { mode: 0o600 });
      fs.appendFileSync(path.join(dir, 'chunks.jsonl'), JSON.stringify({ seq, file: name, startSample, samples,
        threshold: Number(meta.threshold), receivedAt: new Date().toISOString() }) + '\n', { mode: 0o600 });
      return { sessionId: meta.sessionId, seq, samples };
    },
    report(id) { return headReport(sessionDir(id)); },
    clip(id, start, samples) { return headClip(sessionDir(id), start, samples, LiveError); },
    // All-day mode: a few seconds around each detection or "Missed it" press, never continuous audio.
    snippet(bytes, meta, { userAgent } = {}) {
      const samples = parseLiveWav(bytes);
      if (samples > MAX_SNIPPET_SAMPLES) throw new LiveError(413, 'Snippet longer than 8 s');
      if (!SNIPPET_KINDS.includes(meta.kind)) throw new LiveError(400, 'kind must be detection or missed');
      if (!Number.isInteger(meta.seq) || meta.seq < 0 || meta.seq > 100000) throw new LiveError(400, 'seq out of range');
      if (!Number.isInteger(meta.atSample) || meta.atSample < 0) throw new LiveError(400, 'atSample must be a sample index');
      if (usage() + bytes.byteLength > maxBytes) throw new LiveError(507, 'Live store is full');
      const dir = sessionDir(meta.sessionId, true);
      this.ensureMeta(dir, meta, userAgent, 'snippets');
      const name = `snippet-${String(meta.seq).padStart(5, '0')}.wav`;
      const target = path.join(dir, name);
      if (fs.existsSync(target)) throw new LiveError(409, 'Snippet already saved');
      fs.writeFileSync(target, bytes, { mode: 0o600 });
      const row = { seq: meta.seq, file: name, kind: meta.kind, atSample: meta.atSample, startSample: Number(meta.startSample) || 0, samples,
        model: meta.model ? clean(meta.model, 20) : null, score: Number.isFinite(meta.score) ? meta.score : null,
        threshold: Number(meta.threshold), listenedSeconds: Number(meta.listenedSeconds) || 0, receivedAt: new Date().toISOString() };
      fs.appendFileSync(path.join(dir, 'snippets.jsonl'), JSON.stringify(row) + '\n', { mode: 0o600 });
      return row;
    },
    heartbeat(meta, { userAgent } = {}) {
      const dir = sessionDir(meta.sessionId, true);
      this.ensureMeta(dir, meta, userAgent, 'snippets');
      const row = { listenedSeconds: Number(meta.listenedSeconds) || 0, pausedSeconds: Number(meta.pausedSeconds) || 0,
        detections: Number(meta.detections) || 0, threshold: Number(meta.threshold), at: new Date().toISOString() };
      fs.writeFileSync(path.join(dir, 'listening.json'), JSON.stringify(row, null, 2), { mode: 0o600 });
      return row;
    },
    ensureMeta(dir, meta, userAgent, mode) {
      const metaPath = path.join(dir, 'meta.json');
      if (fs.existsSync(metaPath)) return;
      fs.writeFileSync(metaPath, JSON.stringify({
        sessionId: meta.sessionId, startedAt: new Date().toISOString(), inputKind: 'microphone', role: 'owner-live-detector-test', mode,
        release: clean(meta.release, 80), threshold: Number(meta.threshold), userAgent: clean(userAgent, 300),
        humanReview: 'pending', trainingEligible: false,
        note: 'All-day detector test: only short snippets around detections and missed-it presses are stored, plus listening time. Evaluation only.',
      }, null, 2), { mode: 0o600 });
    },
    mark(meta) {
      if (!MARKS.includes(meta.kind)) throw new LiveError(400, 'kind must be missed, false-trigger or correct');
      if (!Number.isInteger(meta.atSample) || meta.atSample < 0) throw new LiveError(400, 'atSample must be a sample index');
      const dir = sessionDir(meta.sessionId, true);
      const row = { kind: meta.kind, atSample: meta.atSample, detection: meta.detection ? clean(meta.detection, 161) : null, at: new Date().toISOString() };
      fs.appendFileSync(path.join(dir, 'marks.jsonl'), JSON.stringify(row) + '\n', { mode: 0o600 });
      return row;
    },
  };
}
