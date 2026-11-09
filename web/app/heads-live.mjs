import fs from 'node:fs';
import path from 'node:path';

const KEY = /^[a-z0-9][a-z0-9_.-]{0,79}::[a-z0-9][a-z0-9_.-]{0,79}$/i;
const unit = x => Number.isFinite(x) && x >= 0 && x <= 1;
export function headMetadata(meta, samples, ErrorType) {
  if (meta.mode !== 'heads') return null;
  const bad = message => { throw new ErrorType(400, message); };
  if (!Array.isArray(meta.heads) || !meta.heads.length || meta.heads.length > 512 ||
      meta.heads.some(key => typeof key !== 'string' || !KEY.test(key)) || new Set(meta.heads).size !== meta.heads.length) bad('Invalid release-qualified heads');
  if (!['negative-only', 'diagnostic'].includes(meta.intent)) bad('Invalid recording intent');
  if (meta.modelHashes && (Object.keys(meta.modelHashes).length !== meta.heads.length ||
      meta.heads.some(key => !/^[a-f0-9]{64}$/.test(meta.modelHashes[key] ?? '')))) bad('Invalid classifier hashes');
  if (!Number.isInteger(meta.processedSamples) || meta.processedSamples < 0 || meta.processedSamples > meta.startSample + samples) bad('Invalid processed sample watermark');
  if (!Array.isArray(meta.headFrames) || meta.headFrames.length > 140) bad('Invalid head frames');
  let previous = -1;
  for (const row of meta.headFrames) {
    if (!Number.isInteger(row.f) || row.f <= previous || row.f * 1280 < meta.startSample ||
        (row.f + 1) * 1280 > Math.min(meta.startSample + samples, meta.processedSamples)) bad('Head frame outside its audio chunk');
    if (!Array.isArray(row.s) || !Array.isArray(row.t) || row.s.length !== meta.heads.length || row.t.length !== meta.heads.length ||
        !row.s.every(unit) || !row.t.every(unit)) bad('Head scores and thresholds must cover every head');
    if (!Array.isArray(row.d) || row.d.some(i => !Number.isInteger(i) || i < 0 || i >= meta.heads.length) || new Set(row.d).size !== row.d.length) bad('Invalid crossing indices');
    previous = row.f;
  }
  return { format: 'waqy-head-recording/1', mode: 'heads', sampleRate: 16000, chunkSeconds: 8, heads: meta.heads, intent: meta.intent, modelHashes: meta.modelHashes };
}

export function checkHeadSession(dir, head, ErrorType) {
  if (!head || !fs.existsSync(path.join(dir, 'meta.json'))) return;
  const previous = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  if (previous.mode !== 'heads' || JSON.stringify(previous.heads) !== JSON.stringify(head.heads) || previous.intent !== head.intent || JSON.stringify(previous.modelHashes) !== JSON.stringify(head.modelHashes)) throw new ErrorType(409, 'Recording configuration changed');
}

// chunks.jsonl is the commit record. A retry repairs incomplete writes, never acknowledges
// an audio file alone as a successful surgical log.
export function saveHeadChunk(dir, bytes, meta, samples, ErrorType) {
  const file = `chunk-${String(meta.seq).padStart(6, '0')}.wav`, headScores = file.replace('.wav', '.scores.json');
  const frames = JSON.stringify(meta.headFrames), target = path.join(dir, file), scoreFile = path.join(dir, headScores);
  const existing = headReport(dir).chunks.find(row => row.seq === meta.seq);
  const sameAudio = fs.existsSync(target) && Buffer.from(bytes).equals(fs.readFileSync(target));
  if (existing) {
    const same = sameAudio && fs.existsSync(scoreFile) && fs.readFileSync(scoreFile, 'utf8') === frames &&
      existing.startSample === meta.startSample && existing.processedSamples === meta.processedSamples;
    throw new ErrorType(same ? 409 : 503, same ? 'Chunk already saved' : 'Saved chunk is incomplete or conflicting; keep this tab open');
  }
  if (fs.existsSync(target) && !sameAudio) throw new ErrorType(409, 'Chunk conflicts with existing audio');
  fs.writeFileSync(scoreFile + '.tmp', frames, { mode: 0o600 }); fs.renameSync(scoreFile + '.tmp', scoreFile);
  fs.writeFileSync(target + '.tmp', bytes, { mode: 0o600 }); fs.renameSync(target + '.tmp', target);
  fs.appendFileSync(path.join(dir, 'chunks.jsonl'), JSON.stringify({ seq: meta.seq, file, startSample: meta.startSample, samples,
    headScores, processedSamples: meta.processedSamples, receivedAt: new Date().toISOString() }) + '\n', { mode: 0o600 });
  return { sessionId: meta.sessionId, seq: meta.seq, samples };
}

export function headReport(dir) {
  const readLines = file => fs.existsSync(path.join(dir, file)) ? fs.readFileSync(path.join(dir, file), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
  return { meta: JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')),
    chunks: readLines('chunks.jsonl'), marks: readLines('marks.jsonl') };
}

// Read a bounded interval, crossing any number of chunk boundaries without filling missing audio.
export function headClip(dir, startSample, samples, ErrorType) {
  if (!Number.isInteger(startSample) || startSample < 0 || !Number.isInteger(samples) || samples < 1 || samples > 16000 * 10) throw new ErrorType(400, 'Clip must be 1 sample to 10 seconds');
  const { chunks } = headReport(dir), end = startSample + samples;
  const pcm = [], sorted = chunks.sort((a, b) => a.startSample - b.startSample);
  let at = startSample, header;
  for (const chunk of sorted) {
    if (chunk.startSample > at) break;
    if (chunk.startSample + chunk.samples <= at) continue;
    const bytes = fs.readFileSync(path.join(dir, chunk.file));
    header ??= Buffer.from(bytes.subarray(0, 44));
    const takeEnd = Math.min(end, chunk.startSample + chunk.samples);
    pcm.push(bytes.subarray(44 + (at - chunk.startSample) * 2, 44 + (takeEnd - chunk.startSample) * 2));
    at = takeEnd;
    if (at === end) break;
  }
  if (at !== end) throw new ErrorType(409, 'Audio interval is not completely saved; retry saving first');
  const result = Buffer.concat([header, ...pcm]);
  result.writeUInt32LE(result.length - 8, 4); result.writeUInt32LE(samples * 2, 40);
  return result;
}
