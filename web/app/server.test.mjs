import { test, expect, beforeEach } from 'bun:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHandler, ROOT, PUBLIC_ORIGIN, ALLOWED_HOSTS, digest, verifiedAssets } from './server.mjs';
import { createRecordingStore, parseWav } from './recordings.mjs';

const assets = new Map([['/', { file: path.join(ROOT, 'index.html'), type: 'text/html' }]]);
const speech = { languages: [{ id: 'de', groups: [{ phrases: [{ line: 'ok-to', label_kind: 'positive' }] }] }] };
let root, store, handler;
beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'wake-collect-test-'));
  store = createRecordingStore({ root: path.join(root, 'recordings'), maxCount: 3 });
  handler = createHandler({ assets, speech, store });
});
const request = (pathname, init = {}, host = 'localhost') => handler(new Request('http://' + host + pathname, init));
const same = { origin: PUBLIC_ORIGIN, 'sec-fetch-site': 'same-origin' };
const SESSION = '0f8fad5b-d9cb-469f-a165-70867728950e';
function wav(samples = 16000) {
  const buffer = new ArrayBuffer(44 + samples * 2), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples * 0.8; i++) view.setInt16(44 + i * 2, Math.round(Math.sin(i / 9) * 6000), true);
  return new Uint8Array(buffer);
}
const meta = () => ({ sessionId: SESSION, language: 'de', group: 'de-primary', label: 'positive', take: 1,
  phrase: 'ok-to', spoken: 'ok-to', words: '2 syllables, normal pace' });
function upload(bytes = wav(), fields = meta(), headers = same) {
  const body = new FormData();
  body.set('audio', new Blob([bytes], { type: 'audio/wav' }), 'take.wav');
  body.set('meta', JSON.stringify(fields));
  return request('/api/recordings', { method: 'POST', headers, body });
}

test('the real page asset map serves every module the recorder imports', () => {
  const { assets: real } = verifiedAssets();
  for (const url of ['/', '/app.mjs', '/capture.mjs', '/queue.mjs']) expect(real.has(url)).toBe(true);
});

test('host, origin and route gates; assets stay GET-only with strict public-page CSP', async () => {
  const page = await request('/');
  const policy = page.headers.get('Content-Security-Policy');
  expect(policy).not.toContain("'unsafe-eval'");
  expect(policy).not.toContain("'unsafe-inline'");
  expect(policy).toContain("connect-src 'self'");
  expect(page.headers.get('Cross-Origin-Opener-Policy')).toBe('same-origin');
  expect(page.headers.get('Cross-Origin-Embedder-Policy')).toBe('require-corp');
  expect(page.headers.get('Cache-Control')).toBe('no-store');
  expect(page.headers.get('Permissions-Policy')).toContain('microphone=(self)');
  for (const host of ['unknown.example.net', 'unknown.example.net:443']) {
    expect((await request('/', {}, host)).status).toBe(404);
    const body = new FormData(); body.set('audio', new Blob([wav()], { type: 'audio/wav' }), 'take.wav'); body.set('meta', JSON.stringify(meta()));
    expect((await request('/api/recordings', { method: 'POST', headers: same, body }, host)).status).toBe(404);
  }
  expect(await (await request('/')).text()).toContain('Takes stay privately on this host');
  expect(await (await request('/')).text()).toContain('Saving is not consent to training');
  for (const method of ['POST', 'PUT', 'DELETE', 'PATCH']) expect((await request('/app.mjs', { method, headers: same })).status).toBe(405);
  for (const pathname of ['/server.mjs', '/recordings.mjs', '/../etc/passwd', '/%2F..%2Fetc', '/?y=1']) expect((await request(pathname)).status).toBe(404);
  // No Content-Type form means the form gate replies 415, not a 405 route claim.
  expect((await request('/api/recordings', { method: 'POST', headers: same })).status).toBe(415);
});

test('saves a real take with all collection fields, then lists and deletes it', async () => {
  const bytes = wav();
  const created = await upload(bytes);
  expect(created.status).toBe(201);
  const { id, durationSeconds } = await created.json();
  expect(id).toMatch(/^\d{8}T\d{6}Z-[0-9a-f]{8}$/);
  expect(durationSeconds).toBe(1);
  const dir = path.join(store.root, id);
  expect(new Uint8Array(fs.readFileSync(path.join(dir, 'audio.wav')))).toEqual(bytes);
  expect(fs.statSync(store.root).mode & 0o777).toBe(0o700);
  expect(fs.statSync(path.join(dir, 'audio.wav')).mode & 0o777).toBe(0o600);
  const row = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  expect(row).toMatchObject({ id, sessionId: SESSION, language: 'de', group: 'de-primary', label: 'positive', take: 1,
    phrase: 'ok-to', spoken: 'ok-to', role: 'owner-wake-candidate', humanReview: 'pending', trainingEligible: false });
  expect(row.audio.peakPcm).toBeGreaterThan(0);
  const listing = await (await request('/api/recordings', { method: 'GET', headers: same })).json();
  expect(listing.recordings.map(row => row.id)).toEqual([id]);
  expect((await request('/api/recordings', { method: 'GET' })).status).toBe(200);
  const summary = await (await request('/api/recordings/summary', { method: 'GET', headers: same })).json();
  expect(summary.total).toBe(1);
  expect(summary.byWord).toEqual({});
  expect(summary.byLabel).toEqual({ positive: 1 });
  const listed = await (await request(`/api/recordings/${id}`, { headers: same })).json();
  expect(listed.recording).toMatchObject({ id, phrase: 'ok-to', label: 'positive', take: 1 });
  expect((await request(`/api/recordings/${id}`, { method: 'DELETE', headers: same })).status).toBe(200);
  expect(fs.existsSync(dir)).toBe(false);
  expect((await request(`/api/recordings/${id}`, { method: 'DELETE', headers: same })).status).toBe(404);
});

test('coverage summary counts the whole store, not the capped listing', async () => {
  // This fixture needs more than the 3-take default cap.
  const big = createRecordingStore({ root: path.join(root, 'big'), maxCount: 100 });
  handler = createHandler({ assets, speech, store: big });
  const meta = (n) => ({ sessionId: crypto.randomUUID(), language: 'de', group: 'bare', label: 'positive',
    take: n, word: 'hex', form: 'bare', phrase: `hex take ${n}`, spoken: 'Hex', words: '' });
  for (let n = 1; n <= 33; n++) expect((await upload(wav(), meta(n))).status).toBe(201);
  const listing = await (await request('/api/recordings', { method: 'GET', headers: same })).json();
  expect(listing.recordings.length).toBe(30);
  const summary = await (await request('/api/recordings/summary', { headers: same })).json();
  expect(summary.total).toBe(33);
  expect(summary.byWord.hex.positive).toBe(33);
  expect(summary.byWord.hex.negative).toBe(0);
  expect(summary.byWord.hex.languages).toEqual({ de: 33 });
  expect(summary.byForm).toEqual({ bare: 33 });
  // Per-phrase counts feed the resumable queue; style is part of the key (missing style = normal).
  expect(summary.byPhrase['hex|de|positive|normal|hex take 7']).toBe(1);
  for (const style of ['normal', 'whispered']) expect((await upload(wav(), { ...meta(7), style })).status).toBe(201);
  const again = await (await request('/api/recordings/summary', { headers: same })).json();
  expect(again.byPhrase['hex|de|positive|normal|hex take 7']).toBe(2);
  expect(again.byPhrase['hex|de|positive|whispered|hex take 7']).toBe(1);
});

test('takes can come from several mics at once; the summary counts per mic', async () => {
  // The fixture store caps at 3 takes: two mics, then one take without a mic label (older pages).
  expect((await upload(wav(), { ...meta(), mic: 'MacBook Pro Microphone', micDevice: 'builtin-1' })).status).toBe(201);
  expect((await upload(wav(), { ...meta(), mic: 'USB Audio', micDevice: 'usb-1' })).status).toBe(201);
  expect((await upload(wav(), meta())).status).toBe(201);
  const summary = await (await request('/api/recordings/summary', { headers: same })).json();
  expect(summary.byMic).toEqual({ 'MacBook Pro Microphone': 1, 'USB Audio': 1 });
  expect((await (await request('/api/recordings', { headers: same })).json()).recordings.every(row => 'mic' in row)).toBe(true);
});

test('review routes serve one take and refuse unknown ids', async () => {
  const created = await (await upload(wav(), { ...meta(), word: 'hex', form: 'bare' })).json();
  const audio = await request(`/api/recordings/${created.id}/audio`, { headers: same });
  expect(audio.status).toBe(200);
  expect(audio.headers.get('Content-Type')).toBe('audio/wav');
  expect(audio.headers.get('Cache-Control')).toBe('no-store');
  expect(new Uint8Array(await audio.arrayBuffer())).toEqual(wav());
  expect((await request('/api/recordings/20260101T000000Z-deadbeef/audio')).status).toBe(404);
  expect((await request('/api/recordings/..%2F..%2Fetc%2Fpasswd/audio')).status).toBe(404);
  const takes = await (await request('/api/recordings/takes/hex', { headers: same })).json();
  expect(takes.takes.map(t => t.id)).toEqual([created.id]);
  expect(takes.takes[0]).toMatchObject({ phrase: 'ok-to', label: 'positive', form: 'bare' });
  expect((await (await request('/api/recordings/takes/..%2Fetc', { headers: same })).json()).takes).toEqual([]);
  expect((await request(`/api/recordings/${created.id}`, { method: 'DELETE', headers: same })).status).toBe(200);
  expect((await request(`/api/recordings/${created.id}/audio`)).status).toBe(404);
});

test('refuses foreign/cross-site origins, silent bytes, non-WAV, long takes, incomplete meta, unknown ids', async () => {
  expect((await upload(wav(), meta(), { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site' })).status).toBe(403);
  expect((await upload(wav(), meta(), { origin: PUBLIC_ORIGIN, 'sec-fetch-site': 'cross-site' })).status).toBe(403);
  expect((await upload(wav(), meta(), {})).status).toBe(403);
  const zeros = wav(); const zv = new DataView(zeros.buffer); for (let i = 44; i < zeros.byteLength; i += 2) zv.setInt16(i, 0, true);
  expect((await upload(zeros)).status).toBe(415);
  expect((await upload(new TextEncoder().encode('plain text, not wav bytes padded to size'))).status).toBe(413);
  expect((await upload(wav(16000 * 6))).status).toBe(413);
  expect((await upload(wav(), { ...meta(), sessionId: 'nope' })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), language: 'fr' })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), label: 'maybe' })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), take: 'x' })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), group: '' })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), take: 0 })).status).toBe(400);
  expect((await upload(wav(), { ...meta(), phrase: '' })).status).toBe(400);
  for (const id of ['..', '..%2F..', 'meta.json', '20261021T000000Z-deadbeef']) {
    expect([404, 405]).toContain((await request(`/api/recordings/${id}`, { method: 'DELETE', headers: same })).status);
  }
  // Every earlier upload in this test was rejected, so the store is empty: fill the
  // 3-record cap from clean state, then prove overflow and deletion.
  expect(fs.readdirSync(store.root)).toHaveLength(0);
  for (const take of [1, 2, 3]) expect((await upload(wav(), { ...meta(), take })).status).toBe(201);
  expect((await upload(wav(), { ...meta(), take: 4 })).status).toBe(507);
  expect(fs.readdirSync(store.root)).toHaveLength(3);
  for (const id of fs.readdirSync(store.root)) {
    expect((await request(`/api/recordings/${id}`, { method: 'DELETE', headers: same })).status).toBe(200);
  }
  expect(fs.readdirSync(store.root)).toHaveLength(0);

});


test('parseWav accepts exact PCM16 mono16k takes and rejects doubled/louder tails', async () => {
  const short = new Uint8Array(wav(800));
  expect(parseWav(short).durationSeconds).toBe(0.05);
  const padded = new Uint8Array(wav());
  expect(parseWav(padded).peakPcm).toBeGreaterThan(0);
});

test('detector routes: sha-verified models, pinned ORT, wasm-unsafe-eval only on the worker', async () => {
  const { detectorAssets, WORKER_CSP } = await import('./server.mjs');
  const { createHash } = await import('node:crypto');
  const models = fs.mkdtempSync(path.join(os.tmpdir(), 'wake-models-'));
  const bytes = new Uint8Array([8, 1, 2, 3]);
  for (const name of ['mel.onnx', 'emb.onnx', 'hey.onnx']) fs.writeFileSync(path.join(models, name), bytes);
  const sha = createHash('sha256').update(bytes).digest('hex');
  const manifest = { threshold: 0.5, classifiers: ['hey_oqto'], description: 'test',
    files: { mel: { file: 'mel.onnx', sha256: sha }, embedding: { file: 'emb.onnx', sha256: sha }, hey_oqto: { file: 'hey.onnx', sha256: sha } } };
  fs.writeFileSync(path.join(models, 'manifest.json'), JSON.stringify(manifest));
  const webDir = path.join(process.env.HOME, 'byteowlz/waqy/web');
  const { assets: detector } = detectorAssets({ webDir, modelsDir: models });
  const all = new Map([...assets, ...detector]);
  const h = createHandler({ assets: all, speech, store });
  const get = p => h(new Request('http://localhost' + p));
  const worker = await get('/detect-worker.mjs');
  expect(worker.status).toBe(200);
  expect(worker.headers.get('Content-Security-Policy')).toBe(WORKER_CSP);
  expect(WORKER_CSP).toContain("'wasm-unsafe-eval'");
  for (const p of ['/', '/detect', '/detect.mjs']) expect((await get(p)).headers.get('Content-Security-Policy')).not.toContain('wasm-unsafe-eval');
  const wasm = await get('/ort/ort-wasm-simd-threaded.wasm');
  expect(wasm.headers.get('Content-Type')).toBe('application/wasm');
  expect((await get('/models/hey.onnx')).status).toBe(200);
  expect((await get('/models/manifest.json')).status).toBe(200);
  expect((await get('/models/other.onnx')).status).toBe(404);
  fs.writeFileSync(path.join(models, 'hey.onnx'), new Uint8Array([9]));
  expect(() => detectorAssets({ webDir, modelsDir: models })).toThrow('checksum mismatch');
});

test('live recording: chunks + scores + marks saved privately; origin, format and size gates', async () => {
  const { createLiveStore } = await import('./live.mjs');
  const liveRoot = path.join(root, 'live');
  const live = createLiveStore({ root: liveRoot, maxBytes: 4 * 1024 * 1024 });
  const h = createHandler({ assets, speech, store, live });
  const post = (p, body, headers = same) => h(new Request('http://localhost' + p, { method: 'POST', headers, body }));
  const chunk = (seq, bytes = wav(16000 * 10), extra = {}) => {
    const body = new FormData();
    body.set('audio', new Blob([bytes], { type: 'audio/wav' }), 'chunk.wav');
    body.set('meta', JSON.stringify({ sessionId: SESSION, seq, startSample: seq * 160000, release: 'r1', threshold: 0.5,
      scores: [{ f: 3, s: { hey_oqto: 0.61, oqto: 0.1 }, d: ['hey_oqto'] }], ...extra }));
    return body;
  };
  expect((await post('/api/live/chunk', chunk(0))).status).toBe(201);
  expect((await post('/api/live/chunk', chunk(0))).status).toBe(409);
  expect((await post('/api/live/chunk', chunk(1), { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site' })).status).toBe(403);
  expect((await post('/api/live/chunk', chunk(1, new Uint8Array([1, 2, 3])))).status).toBe(415);
  expect((await post('/api/live/chunk', chunk(1, wav(16000 * 12)))).status).toBe(413);
  expect((await post('/api/live/chunk', chunk(1, wav(16000), { sessionId: '../x' }))).status).toBe(400);
  const silent = wav(16000); silent.fill(0, 44);
  expect((await post('/api/live/chunk', chunk(1, silent))).status).toBe(201); // quiet room audio is real data
  const markOk = await post('/api/live/mark', JSON.stringify({ sessionId: SESSION, kind: 'missed', atSample: 12345 }), { ...same, 'content-type': 'application/json' });
  expect(markOk.status).toBe(201);
  expect((await post('/api/live/mark', JSON.stringify({ sessionId: SESSION, kind: 'delete-all', atSample: 1 }), same)).status).toBe(400);
  expect((await h(new Request('http://localhost/api/live/chunk'))).status).toBe(403);
  expect((await h(new Request('http://localhost/api/live/chunk', { headers: same }))).status).toBe(405);
  const dir = path.join(liveRoot, SESSION);
  expect(fs.readdirSync(dir).sort()).toEqual(['chunk-000000.wav', 'chunk-000001.wav', 'chunks.jsonl', 'marks.jsonl', 'meta.json', 'scores.jsonl']);
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  expect(meta.trainingEligible).toBe(false);
  expect(fs.readFileSync(path.join(dir, 'scores.jsonl'), 'utf8')).toContain('"hey_oqto":0.61');
  expect(fs.statSync(dir).mode & 0o777).toBe(0o700);
  const off = createHandler({ assets, speech, store });
  expect((await off(new Request('http://localhost/api/live/chunk', { method: 'POST', headers: same, body: chunk(5) }))).status).toBe(404);
});

test('takes carry a speaking style; old pages default to normal, unknown styles are refused', async () => {
  const saved = await upload(wav(), { ...meta(), style: 'whispered' });
  expect(saved.status).toBe(201);
  const { id } = await saved.json();
  expect(JSON.parse(fs.readFileSync(path.join(root, 'recordings', id, 'meta.json'), 'utf8')).style).toBe('whispered');
  const old = await upload(wav(), meta());
  const oldId = (await old.json()).id;
  expect(JSON.parse(fs.readFileSync(path.join(root, 'recordings', oldId, 'meta.json'), 'utf8')).style).toBe('normal');
  expect((await upload(wav(), { ...meta(), style: 'shouted' })).status).toBe(400);
});

test('all-day mode stores only snippets + listening time; long snippets and unknown kinds refused', async () => {
  const { createLiveStore } = await import('./live.mjs');
  const liveRoot = path.join(root, 'live2');
  const h = createHandler({ assets, speech, store, live: createLiveStore({ root: liveRoot, maxBytes: 4 * 1024 * 1024 }) });
  const post = (p, body, headers = same) => h(new Request('http://localhost' + p, { method: 'POST', headers, body }));
  const snip = (seq, bytes = wav(16000 * 4), extra = {}) => {
    const body = new FormData();
    body.set('audio', new Blob([bytes], { type: 'audio/wav' }), 's.wav');
    body.set('meta', JSON.stringify({ sessionId: SESSION, seq, kind: 'detection', atSample: 99000, startSample: 50000, model: 'oqto', score: 0.71, threshold: 0.5, listenedSeconds: 600, release: 'r', ...extra }));
    return body;
  };
  expect((await post('/api/live/snippet', snip(0))).status).toBe(201);
  expect((await post('/api/live/snippet', snip(0))).status).toBe(409);
  expect((await post('/api/live/snippet', snip(1, wav(16000 * 9)))).status).toBe(413);
  expect((await post('/api/live/snippet', snip(2, wav(16000), { kind: 'everything' }))).status).toBe(400);
  expect((await post('/api/live/snippet', snip(3), { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site' })).status).toBe(403);
  expect((await post('/api/live/heartbeat', JSON.stringify({ sessionId: SESSION, listenedSeconds: 3600, detections: 2, threshold: 0.5 }))).status).toBe(201);
  const dir = path.join(liveRoot, SESSION);
  expect(fs.readdirSync(dir).sort()).toEqual(['listening.json', 'meta.json', 'snippet-00000.wav', 'snippets.jsonl']);
  expect(JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')).mode).toBe('snippets');
  expect(JSON.parse(fs.readFileSync(path.join(dir, 'listening.json'), 'utf8')).listenedSeconds).toBe(3600);
});

// ---------------------------------------------------------------- head explorer
import { packsAssets, headsAssets } from './server.mjs';

function fakePack(root, name, heads = 2) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  const files = {};
  const write = (id, file, bytes) => {
    fs.writeFileSync(path.join(dir, file), bytes);
    files[id] = { sha256: digest(path.join(dir, file)), file };
  };
  write('mel', 'melspectrogram_raw.onnx', 'mel');
  write('embedding', 'embedding_model.onnx', 'embed');
  const classifiers = [];
  for (let i = 0; i < heads; i++) {
    classifiers.push(`head_${i}`);
    write(`head_${i}`, `head_${i}.onnx`, `w${i}`);
  }
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify({ release: name, threshold: 0.6, classifiers, files }));
  return { dir, files };
}

test('packs: releases are listed with their own manifest and threshold', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-packs-'));
  fakePack(root, 'rel-a', 3);
  fakePack(root, 'rel-b', 2);
  fs.mkdirSync(path.join(root, 'not-a-release'));
  const { assets, releases } = packsAssets({ packsDir: root });
  expect(releases.map((r) => r.name)).toEqual(['rel-a', 'rel-b']);
  expect(releases[0].classifiers.length).toBe(3);
  expect(releases[0].threshold).toBe(0.6);
  expect(releases[0].shared_bytes).toBe(8);
  expect(releases[0].head_bytes).toEqual({ head_0: 2, head_1: 2, head_2: 2 });
  expect(assets.has('/packs/rel-a/head_0.onnx')).toBe(true);
  expect(assets.has('/packs/rel-b/manifest.json')).toBe(true);
  expect(assets.has('/packs/not-a-release/manifest.json')).toBe(false);
});

test('packs: a release whose bytes do not match its manifest is not offered', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-packs-'));
  const { dir } = fakePack(root, 'rel-tampered', 2);
  fs.writeFileSync(path.join(dir, 'head_1.onnx'), 'evil');
  const { assets, releases } = packsAssets({ packsDir: root });
  expect(releases).toEqual([]);
  expect(assets.size).toBe(0);
});

test('packs: names that could escape the directory are ignored', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-packs-'));
  fakePack(root, '..evil', 1);
  expect(packsAssets({ packsDir: root }).releases).toEqual([]);
});

test('head explorer page, client and worker are served, worker gets the wasm CSP', () => {
  const { assets } = headsAssets({ webDir: path.join(process.env.HOME, 'byteowlz/waqy/web') });
  for (const url of ['/heads', '/heads.mjs', '/heads.css', '/heads-state.mjs', '/heads-capture.mjs', '/heads-worker.mjs', '/waqy-engine.mjs', '/ort/ort.wasm.min.mjs']) {
    expect(assets.has(url)).toBe(true);
  }
  expect(assets.get('/heads-worker.mjs').csp).toMatch(/wasm-unsafe-eval/);
  expect(assets.get('/heads.mjs').type).toMatch(/javascript/);
});

test('handler exposes /api/releases and refuses paths outside an offered pack', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-packs-'));
  fakePack(root, 'rel-a', 1);
  const { assets, releases } = packsAssets({ packsDir: root });
  const handler = createHandler({ assets, speech: { phrases: [] }, store: null, releases });
  const list = await handler(new Request('http://127.0.0.1:8036/api/releases'));
  expect(list.status).toBe(200);
  expect((await list.json()).releases[0].name).toBe('rel-a');
  expect((await handler(new Request('http://127.0.0.1:8036/packs/rel-a/head_0.onnx'))).status).toBe(200);
  expect((await handler(new Request('http://127.0.0.1:8036/packs/rel-a/../manifest.json'))).status).not.toBe(200);
  expect((await handler(new Request('http://127.0.0.1:8036/packs/rel-zzz/manifest.json'))).status).toBe(404);
});
