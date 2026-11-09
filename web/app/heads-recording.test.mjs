import { test, expect } from 'bun:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { HeadRecording, RECORD_CHUNK } from './heads-recording.mjs';
import { createLiveStore } from './live.mjs';
import { createHandler, PUBLIC_ORIGIN } from './server.mjs';

function wav(pcm) {
  const buffer = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buffer);
  const tag = (at, s) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  tag(0, 'RIFF'); v.setUint32(4, buffer.byteLength - 8, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); tag(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + 2 * i, pcm[i] * 32767, true);
  return new Blob([buffer], { type: 'audio/wav' });
}
function fixture(count = 66) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-head-record-'));
  const live = createLiveStore({ root });
  const handler = createHandler({ assets: new Map(), live });
  const send = (route, options) => handler(new Request('http://localhost' + route, {
    ...options, headers: { origin: PUBLIC_ORIGIN, 'sec-fetch-site': 'same-origin', ...options.headers },
  }));
  const heads = Array.from({ length: count }, (_, i) => `release-long-name::head_${i}`);
  const board = new Map(heads.map(key => [key, { threshold: 0.5 }]));
  const record = new HeadRecording({ heads, wav, send });
  const feed = (frame, score = 0.75) => {
    record.push(new Float32Array(1280).fill(frame % 2 ? 0.25 : -0.25));
    record.observe({ results: [{ frame, scores: Object.fromEntries(heads.map(key => [key, score])) }], processedSamples: (frame + 1) * 1280 }, board);
  };
  const post = (route, body) => send('/api/live/' + route, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return { root, live, handler, heads, board, record, feed, post };
}

test('long all-head recording preserves scores/thresholds and frees uploaded audio, without 4-head truncation', async () => {
  const x = fixture();
  try {
    for (let chunk = 0; chunk < 5; chunk++) {
      for (let i = 0; i < 100; i++) x.feed(chunk * 100 + i, i === 0 ? 0.25 : 0.75);
      await x.record.queue;
      expect(x.record.parts.length).toBe(0); expect(x.record.frames.length).toBe(0);
    }
    x.feed(500); await x.record.finish();
    expect(x.record.samples).toBe(501 * 1280); expect(x.record.saved).toBe(x.record.samples);
    const dir = path.join(x.root, x.record.sessionId), report = x.live.report(x.record.sessionId);
    expect(report.meta.heads).toEqual(x.heads); expect(report.meta.intent).toBe('negative-only');
    expect(report.meta.trainingEligible).toBe(false); expect(report.chunks.length).toBe(6);
    const rows = JSON.parse(fs.readFileSync(path.join(dir, report.chunks[0].headScores)));
    expect(rows.length).toBe(100); expect(rows[1].s.length).toBe(66); expect(rows[1].t.length).toBe(66); expect(rows[1].d.length).toBe(66);
    expect(report.chunks[5].startSample).toBe(5 * RECORD_CHUNK);
    expect(fs.statSync(dir).mode & 0o777).toBe(0o700);
    const event = x.record.events.at(-1); await x.record.mark(event);
    expect(x.live.report(x.record.sessionId).marks[0].detection).toBe(event.key);
    expect(event.key.length).toBeGreaterThan(20);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('cross-boundary replay is sample-exact; missing audio is refused instead of padded', async () => {
  const x = fixture(2);
  try {
    for (let frame = 0; frame < 101; frame++) x.feed(frame);
    await x.record.finish();
    const response = await x.post('clip', { sessionId: x.record.sessionId, startSample: RECORD_CHUNK - 1280, samples: 2560 });
    expect(response.status).toBe(200);
    const bytes = new DataView(await response.arrayBuffer());
    expect(bytes.getUint32(40, true)).toBe(5120);
    expect(bytes.getInt16(44, true)).toBe(8191); expect(bytes.getInt16(44 + 2560, true)).toBe(-8191);
    expect((await x.post('clip', { sessionId: x.record.sessionId, startSample: RECORD_CHUNK, samples: 2560 })).status).toBe(409);
    expect((await x.post('clip', { sessionId: '../bad', startSample: 0, samples: 2560 })).status).toBe(400);
    expect((await x.handler(new Request('http://localhost/api/live/report', { method: 'POST', body: '{}' }))).status).toBe(403);
    const report = await x.post('report', { sessionId: x.record.sessionId }); expect(report.status).toBe(200);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('score dimension/sample alignment errors reject before writing audio', async () => {
  const x = fixture(2);
  try {
    x.record.send = async (_, { body }) => {
      const meta = JSON.parse(body.get('meta')); meta.headFrames[0].s.pop();
      body.set('meta', JSON.stringify(meta));
      return x.handler(new Request('http://localhost/api/live/chunk', { method: 'POST', headers: { origin: PUBLIC_ORIGIN }, body }));
    };
    x.feed(0); await x.record.finish();
    expect(x.record.failed.length).toBe(1); expect(x.record.saved).toBe(0); expect(fs.readdirSync(x.root)).toEqual([]);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('failed saves remain retryable; backpressure stops rather than silently dropping PCM', async () => {
  const x = fixture(1);
  try {
    const send = x.record.send; x.record.send = async () => new Response('Full', { status: 507 });
    x.feed(0); await x.record.finish(); expect(x.record.failed.length).toBe(1);
    expect(() => x.record.push(new Float32Array(1280))).toThrow();
    x.record.send = send; await x.record.retry(); expect(x.record.failed.length).toBe(0); expect(x.record.saved).toBe(1280);
    x.record.samples = RECORD_CHUNK * 5;
    expect(() => x.record.push(new Float32Array(1280))).toThrow('Saving cannot keep up');
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('cutter groups simultaneous heads and keeps every proposed negative pending review', async () => {
  const x = fixture(6);
  try {
    for (let i = 0; i < 101; i++) x.feed(i, i === 99 ? 0.25 : 0.75);
    await x.record.finish();
    const session = path.join(x.root, x.record.sessionId), out = path.join(x.root, 'cuts');
    const result = Bun.spawnSync(['bun', path.resolve(import.meta.dir, '../../tools/cut-head-recording.mjs'), '--session', session, '--out', out]);
    expect(result.exitCode).toBe(0);
    const rows = fs.readFileSync(path.join(out, 'manifest.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    expect(rows.length).toBe(2); expect(rows[1].atSample).toBe(101 * 1280);
    expect(rows[1].heads.length).toBe(6); expect(rows[1].trainingEligible).toBe(false);
    expect(rows[1].humanReview).toBe('pending'); expect(rows[1].sha256.length).toBe(64);
    expect(fs.existsSync(path.join(out, rows[1].file))).toBe(true);
    expect(Bun.spawnSync(['bun', path.resolve(import.meta.dir, '../../tools/cut-head-recording.mjs'), '--session', session, '--out', out]).exitCode).not.toBe(0);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('an incomplete saved score sidecar is not acknowledged as a successful duplicate', async () => {
  const x = fixture(6);
  try {
    let job;
    const send = x.record.send;
    x.record.send = async (route, options) => { job = { route, options }; return send(route, options); };
    x.feed(0); await x.record.finish();
    expect((await send(job.route, job.options)).status).toBe(409);
    const report = x.live.report(x.record.sessionId);
    fs.unlinkSync(path.join(x.root, x.record.sessionId, report.chunks[0].headScores));
    expect((await send(job.route, job.options)).status).toBe(503);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});

test('threshold edits retain exact per-frame thresholds and do not fabricate a new crossing', async () => {
  const x = fixture(1);
  try {
    x.feed(0, 0.4); x.board.get(x.heads[0]).threshold = 0.3; x.record.threshold(x.heads[0], 0.3); x.feed(1, 0.4);
    x.feed(2, 0.1); x.feed(3, 0.4); await x.record.finish();
    const report = x.live.report(x.record.sessionId), dir = path.join(x.root, x.record.sessionId);
    const rows = JSON.parse(fs.readFileSync(path.join(dir, report.chunks[0].headScores)));
    expect(rows.map(row => row.t[0])).toEqual([0.5, 0.3, 0.3, 0.3]); expect(rows.map(row => row.d.length)).toEqual([0, 0, 0, 1]);
    expect(x.record.events[0].atSample).toBe(5120);
  } finally { fs.rmSync(x.root, { recursive: true, force: true }); }
});
