import assert from 'node:assert';import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRecordingStore } from './recordings.mjs';
import { createLookalikeStore, editorState, effective, LookalikeError } from './lookalikes.mjs';
import { createLiveStore, LiveError } from './live.mjs';

export const ROOT = import.meta.dir;
// Configure for your own host: WAQY_PUBLIC_ORIGIN=https://waqy.example.net
// WAQY_ALLOWED_HOSTS=waqy.example.net,localhost,127.0.0.1
export const PUBLIC_ORIGIN = process.env.WAQY_PUBLIC_ORIGIN ?? 'http://127.0.0.1:8080';
// Extra origins accepted for mutations during a host rename: WAQY_EXTRA_ORIGINS=https://old.example.net
export const EXTRA_ORIGINS = (process.env.WAQY_EXTRA_ORIGINS ?? '').split(',').map((o) => o.trim()).filter(Boolean);
export const ALLOWED_HOSTS = (process.env.WAQY_ALLOWED_HOSTS ?? 'localhost,127.0.0.1')
  .split(',').map((h) => h.trim().toLowerCase()).filter(Boolean);
export const RESAMPLER_SOURCE = path.join(ROOT, 'vendor/resampler.ts');  // vendored, MIT, see vendor/LICENSE.ears-browser
// The vendored resampler must not change silently: it is built from a pinned source.
export const RESAMPLER_SHA = 'f873572a4edf15e6d84d599858fbf6f5aad1c667b49a53403c75faf170f089b4';
export const digest = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const HOME = process.env.HOME ?? '/';

export const securityHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Cache-Control': 'no-store',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'Permissions-Policy': 'microphone=(self), camera=(), geolocation=()',
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; worker-src 'self' blob:; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
};

// The detector worker compiles WebAssembly (onnxruntime-web); only its response gets 'wasm-unsafe-eval'.
export const WORKER_CSP = securityHeaders['Content-Security-Policy'].replace("script-src 'self'", "script-src 'self' 'wasm-unsafe-eval'");
export const DETECT_FILES = { '/detect': 'detect.html', '/detect.mjs': 'detect.mjs', '/detect-worker.mjs': 'detect-worker.mjs' };
export const HEADS_FILES = { '/heads': 'heads.html', '/heads.mjs': 'heads.mjs', '/heads.css': 'heads.css', '/heads-state.mjs': 'heads-state.mjs', '/heads-capture.mjs': 'heads-capture.mjs', '/heads-worker.mjs': 'heads-worker.mjs', '/heads-recording.mjs': 'heads-recording.mjs', '/heads-review.mjs': 'heads-review.mjs' };
const ORT_FILES = ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const typeOf = name => name.endsWith('.css') ? 'text/css' : name.endsWith('.html') ? 'text/html' : name.endsWith('.wasm') ? 'application/wasm'
  : name.endsWith('.json') ? 'application/json' : name.endsWith('.onnx') ? 'application/octet-stream' : 'text/javascript';

/** Detector page assets: engine + pinned onnxruntime-web from webDir, models listed (and sha-verified) in modelsDir/manifest.json. */
export function detectorAssets({ webDir, modelsDir }) {
  const assets = new Map();
  for (const [url, name] of Object.entries(DETECT_FILES)) {
    assets.set(url, { file: path.join(ROOT, name), type: typeOf(name), csp: name === 'detect-worker.mjs' ? WORKER_CSP : undefined });
  }
  assets.set('/waqy-engine.mjs', { file: path.join(webDir, 'waqy-engine.mjs'), type: 'text/javascript' });
  const ortDir = path.join(webDir, 'node_modules/onnxruntime-web/dist');
  for (const name of ORT_FILES) assets.set('/ort/' + name, { file: path.join(ortDir, name), type: typeOf(name) });
  const manifest = JSON.parse(fs.readFileSync(path.join(modelsDir, 'manifest.json'), 'utf8'));
  assets.set('/models/manifest.json', { file: path.join(modelsDir, 'manifest.json'), type: 'application/json' });
  for (const [id, entry] of Object.entries(manifest.files)) {
    if (!/^[A-Za-z0-9_.-]+\.onnx$/.test(entry.file)) throw new Error('Bad model file name for ' + id);
    const file = path.join(modelsDir, entry.file);
    if (digest(file) !== entry.sha256) throw new Error('Model checksum mismatch: ' + entry.file);
    assets.set('/models/' + entry.file, { file, type: typeOf(entry.file) });
  }
  for (const { file } of assets.values()) if (!fs.existsSync(file)) throw new Error('Missing detector asset: ' + file);
  return { assets, manifest };
}

/**
 * Every release under packsDir becomes an explorer entry: /api/releases lists them and
 * /packs/<release>/<file> serves the ONNX. A release is exposed only if every classifier hash in its
 * manifest matches, so a tampered or half-copied pack cannot be exercised by accident.
 */
export function packsAssets({ packsDir }) {
  const all = new Map();
  const releases = [];
  if (!fs.existsSync(packsDir)) return { assets: all, releases };
  for (const entry of fs.readdirSync(packsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(entry.name)) continue;
    const dir = path.join(packsDir, entry.name);
    const manifestPath = path.join(dir, 'manifest.json');
    if (!fs.existsSync(manifestPath)) continue;
    let manifest;
    try { manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')); } catch { continue; }
    if (!Array.isArray(manifest.classifiers) || !manifest.classifiers.length || !manifest.files) continue;
    const assets = new Map();
    assets.set(`/packs/${entry.name}/manifest.json`, { file: manifestPath, type: 'application/json' });
    let complete = true;
    for (const id of [...manifest.classifiers, 'mel', 'embedding']) {
      const file = manifest.files[id]?.file;
      if (!file || !/^[A-Za-z0-9_.-]+\.onnx$/.test(file)) { complete = false; break; }
      const path0 = path.join(dir, file);
      if (!fs.existsSync(path0) || digest(path0) !== manifest.files[id].sha256) { complete = false; break; }
      assets.set(`/packs/${entry.name}/${file}`, { file: path0, type: 'application/octet-stream' });
    }
    if (!complete) { console.error(JSON.stringify({ packSkipped: entry.name, reason: 'checksum or file missing' })); continue; }
    const sizeOf = id => fs.statSync(path.join(dir, manifest.files[id].file)).size;
    releases.push({ name: entry.name, release: manifest.release ?? entry.name, classifiers: manifest.classifiers,
      threshold: manifest.threshold ?? 0.5, heads: manifest.heads ?? {},
      head_bytes: Object.fromEntries(manifest.classifiers.map(id => [id, sizeOf(id)])),
      shared_bytes: sizeOf('mel') + sizeOf('embedding') });
    for (const [url, asset] of assets) all.set(url, asset);
  }
  releases.sort((a, b) => a.name.localeCompare(b.name));
  return { assets: all, releases };
}

export function headsAssets({ webDir }) {
  const assets = new Map();
  for (const [url, name] of Object.entries(HEADS_FILES)) {
    assets.set(url, { file: path.join(ROOT, name), type: typeOf(name), csp: name === 'heads-worker.mjs' ? WORKER_CSP : undefined });
  }
  assets.set('/waqy-engine.mjs', { file: path.join(webDir, 'waqy-engine.mjs'), type: 'text/javascript' });
  const ortDir = path.join(webDir, 'node_modules/onnxruntime-web/dist');
  for (const name of ORT_FILES) assets.set('/ort/' + name, { file: path.join(ortDir, name), type: typeOf(name) });
  for (const { file } of assets.values()) if (!fs.existsSync(file)) throw new Error('Missing head explorer asset: ' + file);
  return { assets };
}

export async function buildResampler() {
  // Rebuild the pinned SDK audio.ts here. Bun's emitted header comment embeds the
  // relative source path, so a build from this private root differs only in that
  // comment from the sealed browser-1 build; the code payload is identical.
  assert.equal(digest(RESAMPLER_SOURCE), RESAMPLER_SHA);
  const build = await Bun.build({ entrypoints: [RESAMPLER_SOURCE], naming: 'resampler.js', outdir: path.join(ROOT, 'build'), target: 'browser', format: 'esm' });
  assert.equal(build.success, true, 'Resampler build failed: ' + JSON.stringify(build.logs));
  const out = path.join(ROOT, 'build/resampler.js'), sealed = path.join(HOME, '.local/share/foxline/moonshine-export/side-by-side-1/build/resampler.js');
  const SEALED_SHA = 'fd38696e8ba12487fa9e393db1a75961ae4436f1ec0b67dd1a6bcaa66da25d17';
  const payload = data => data.split('\n').slice(1).join('\n');
  const sameCode = fs.existsSync(sealed) && digest(sealed) === SEALED_SHA && payload(fs.readFileSync(sealed, 'utf8')) === payload(fs.readFileSync(out, 'utf8'));
  return { source: { path: RESAMPLER_SOURCE, sha256: RESAMPLER_SHA }, built: { path: out, sha256: digest(out), bytes: fs.statSync(out).size },
           sealed: { path: sealed, sha256: SEALED_SHA, exists: fs.existsSync(sealed) },
           codePayloadIdenticalToSealed: sameCode,
           difference: sameCode ? 'none' : 'build header comment only differs by relative source path' };
}

export function verifiedAssets() {
  const files = { '/': 'index.html', '/style.css': 'style.css', '/tokens.css': 'tokens.css', '/app.mjs': 'app.mjs',
                  '/capture.mjs': 'capture.mjs', '/queue.mjs': 'queue.mjs', '/lookalikes': 'lookalikes.html', '/lookalikes-page.mjs': 'lookalikes-page.mjs', '/capture-worklet.js': 'capture-worklet.js', '/resampler.js': 'build/resampler.js' };
  const assets = new Map();
  for (const [url, name] of Object.entries(files)) {
    const file = path.join(ROOT, name);
    if (!fs.existsSync(file)) throw new Error('Missing private asset: ' + name);
    assets.set(url, { file, type: name.endsWith('.css') ? 'text/css' : name.endsWith('.html') ? 'text/html' : 'text/javascript' });
  }
  assets.set('/speech.json', { file: path.join(ROOT, 'speech.json'), type: 'application/json' });
  const tts = path.join(ROOT, 'spike/tts-wake.wav');
  if (fs.existsSync(tts)) assets.set('/spike/tts-wake.wav', { file: tts, type: 'audio/wav' });
  return { assets };
}

const reply = (headers, body, status = 200, type = 'text/plain') =>
  new Response(body, { status, headers: { ...headers, 'Content-Type': type } });
const replyJson = (headers, value, status = 200) => reply(headers, JSON.stringify(value), status, 'application/json');
const RECORDING = /^\/api\/recordings\/([^/]+)$/;

// Look-alike editor API. Reads are open to the page; writes need the page origin.
async function lookalikeRoute({ request, url, headers, sameOrigin, wordpack, lookalikes, store }) {
  if (!lookalikes || !wordpack) return reply(headers, 'Look-alike editing is not enabled', 404);
  const state = () => editorState(wordpack, lookalikes.read(), store ? store.summary().byPhrase : {});
  try {
    if (request.method === 'GET') return url.pathname === '/api/lookalikes' ? replyJson(headers, state()) : exportOverrides(headers, lookalikes);
    if (request.method !== 'POST' || url.pathname !== '/api/lookalikes') return reply(headers, 'Not found', 405);
    if (!sameOrigin) return reply(headers, 'Forbidden origin', 403);
    const body = await request.text();
    if (body.length > 2048) return reply(headers, 'Too large', 413);
    const edit = JSON.parse(body);
    if (edit.op === 'delete-takes') return deleteLookalikeTakes(headers, edit, state, store);
    lookalikes.edit(wordpack, edit, edit.revision);
    return replyJson(headers, state());
  } catch (error) {
    if (error instanceof SyntaxError) return reply(headers, 'Invalid JSON', 400);
    if (!(error instanceof LookalikeError)) {
      console.error(JSON.stringify({ lookalikeError: error.message }));
      return reply(headers, 'Could not update the look-alike list', 500);
    }
    return error.status === 409 ? replyJson(headers, { error: error.message, state: state() }, 409) : replyJson(headers, { error: error.message }, error.status);
  }
}
function exportOverrides(headers, lookalikes) {
  return replyJson({ ...headers, 'Content-Disposition': 'attachment; filename="waqy-lookalike-overrides.json"' }, {
    exportedAt: new Date().toISOString(), ...lookalikes.read(),
    note: 'Owner-reviewed look-alike overrides on top of training/words/*.json. removed = word|language|phrase. Apply to the specs before the next head training; existing heads are unchanged.' });
}
// Only for phrases that are removed or no longer listed: deleting a live look-alike's takes
// belongs to the take review list, one take at a time.
function deleteLookalikeTakes(headers, edit, state, store) {
  const lists = state().words.find(word => word.name === edit.word)?.languages?.[edit.language];
  const allowed = lists && (lists.orphans.some(item => item.line === edit.line) || lists.items.some(item => item.line === edit.line && item.removed));
  if (!allowed) return reply(headers, 'Only takes of removed or unlisted look-alikes can be deleted here', 409);
  const result = store.removeLookalikeTakes({ word: edit.word, language: edit.language, phrase: edit.line });
  return replyJson(headers, { ...state(), deleted: result.deleted });
}

export function createHandler({ assets, speech, store, live = null, releases = [], wordpack = null, lookalikes = null, allowedOrigins = [PUBLIC_ORIGIN] }) {
  return async request => {
    const url = new URL(request.url);
    const host = (request.headers.get('host') ?? url.host).toLowerCase().split(':')[0];
    const headers = { ...securityHeaders };
    const origin = request.headers.get('origin') ?? '';
    const site = request.headers.get('sec-fetch-site');
    if (!ALLOWED_HOSTS.includes(host)) return reply(headers, 'Unknown host', 404);
    if (url.search) return reply(headers, 'No query routes', 404);
    if (url.pathname.startsWith('/api/live/')) {
      if (!live) return reply(headers, 'Live recording is not enabled', 404);
      if (!allowedOrigins.includes(origin) || (site && site !== 'same-origin')) return reply(headers, 'Forbidden origin', 403);
      if (request.method !== 'POST') return reply(headers, 'POST only', 405);
      try {
        if (url.pathname === '/api/live/chunk') {
          if (!(request.headers.get('content-type') ?? '').startsWith('multipart/form-data')) return reply(headers, 'Expected multipart form', 415);
          if (Number(request.headers.get('content-length') ?? 0) > 3 * 1024 * 1024) return reply(headers, 'Too large', 413);
          const form = await request.formData();
          const audio = form.get('audio'), meta = form.get('meta');
          if (!(audio instanceof Blob) || typeof meta !== 'string' || meta.length > 2 * 1024 * 1024) return reply(headers, 'Expected audio and meta fields', 400);
          return replyJson(headers, live.chunk(new Uint8Array(await audio.arrayBuffer()), JSON.parse(meta), { userAgent: request.headers.get('user-agent') }), 201);
        }
        if (url.pathname === '/api/live/report' || url.pathname === '/api/live/clip') {
          const body = await request.text();
          if (body.length > 1024) return reply(headers, 'Too large', 413);
          const meta = JSON.parse(body);
          if (url.pathname.endsWith('/report')) return replyJson(headers, live.report(meta.sessionId));
          return new Response(live.clip(meta.sessionId, meta.startSample, meta.samples), {
            headers: { ...headers, 'Content-Type': 'audio/wav' },
          });
        }
        if (url.pathname === '/api/live/snippet') {
          if (!(request.headers.get('content-type') ?? '').startsWith('multipart/form-data')) return reply(headers, 'Expected multipart form', 415);
          if (Number(request.headers.get('content-length') ?? 0) > 512 * 1024) return reply(headers, 'Too large', 413);
          const form = await request.formData();
          const audio = form.get('audio'), meta = form.get('meta');
          if (!(audio instanceof Blob) || typeof meta !== 'string' || meta.length > 4096) return reply(headers, 'Expected audio and meta fields', 400);
          return replyJson(headers, live.snippet(new Uint8Array(await audio.arrayBuffer()), JSON.parse(meta), { userAgent: request.headers.get('user-agent') }), 201);
        }
        if (url.pathname === '/api/live/heartbeat') {
          const body = await request.text();
          if (body.length > 2048) return reply(headers, 'Too large', 413);
          return replyJson(headers, live.heartbeat(JSON.parse(body), { userAgent: request.headers.get('user-agent') }), 201);
        }
        if (url.pathname === '/api/live/mark') {
          const body = await request.text();
          if (body.length > 2048) return reply(headers, 'Too large', 413);
          return replyJson(headers, live.mark(JSON.parse(body)), 201);
        }
        return reply(headers, 'Not found', 404);
      } catch (error) {
        if (error instanceof SyntaxError) return reply(headers, 'Invalid JSON', 400);
        if (error instanceof LiveError) return reply(headers, error.message, error.status);
        console.error(JSON.stringify({ liveError: error.message }));
        return reply(headers, 'Could not save', 500);
      }
    }
    if (url.pathname === '/api/lookalikes' || url.pathname === '/api/lookalikes/export') {
      return lookalikeRoute({ request, url, headers, sameOrigin: allowedOrigins.includes(origin) && (!site || site === 'same-origin'), wordpack, lookalikes, store });
    }
    if (url.pathname.startsWith('/api/recordings')) {
      // Mutations only from this exact page origin; cross-site or no-Origin refused.
      // Reads are listing/metadata and carry no audio mutation.
      const mutating = request.method !== 'GET';
      if (mutating && (!allowedOrigins.includes(origin) || (site && site !== 'same-origin'))) return reply(headers, 'Forbidden origin', 403);
      try {
        if (url.pathname === '/api/recordings' && request.method === 'POST') {
          if (!(request.headers.get('content-type') ?? '').startsWith('multipart/form-data')) return reply(headers, 'Expected multipart form', 415);
          if (Number(request.headers.get('content-length') ?? 0) > 512 * 1024) return reply(headers, 'Too large', 413);
          const form = await request.formData();
          const audio = form.get('audio'), meta = form.get('meta');
          if (!(audio instanceof Blob) || typeof meta !== 'string' || meta.length > 2048) return reply(headers, 'Expected audio and meta fields', 400);
          const created = store.create(new Uint8Array(await audio.arrayBuffer()), JSON.parse(meta), { userAgent: request.headers.get('user-agent') });
          return replyJson(headers, created, 201);
        }
        if (url.pathname === '/api/recordings') return replyJson(headers, { recordings: store ? store.list() : [] });
        if (url.pathname === '/api/recordings/summary') return replyJson(headers, store ? store.summary() : { total: 0, byWord: {} });
        // Path, not a query string: the handler intentionally refuses all query routes.
        const takes = /^\/api\/recordings\/takes\/([^/]+)$/.exec(url.pathname);
        if (takes) {
          const key = decodeURIComponent(takes[1]);
          return replyJson(headers, { takes: store ? (key === 'all' ? store.forWord(null, 300) : store.forWord(key)) : [] });
        }
        const audio = /^\/api\/recordings\/([^/]+)\/audio$/.exec(url.pathname);
        if (audio) {
          const file = store?.audioPath(audio[1]);
          if (!file) return reply(headers, 'Not found', 404);
          return reply({ ...headers, 'Cache-Control': 'no-store' }, Bun.file(file), 200, 'audio/wav');
        }
        const match = RECORDING.exec(url.pathname);
        if (!match) return reply(headers, url.pathname === '/api/recordings' ? 'Read-only listing' : 'Not found', 405);
        if (request.method === 'GET') return replyJson(headers, { recording: store.list().filter(row => row.id === match[1])[0] ?? null });
        if (request.method === 'DELETE') return replyJson(headers, store.remove(match[1]));
        return reply(headers, 'Not found', 405);
      } catch (error) {
        if (error instanceof SyntaxError) return reply(headers, 'Invalid JSON', 400);
        if (Number.isInteger(error.status)) return reply(headers, error.message, error.status);
        console.error(JSON.stringify({ recordingError: error.message }));
        return reply(headers, 'Could not save recording', 500);
      }
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') return reply(headers, 'Read-only assets', 405);
    if (url.pathname === '/health') return replyJson(headers, { ok: true, service: 'waqy', microphoneRecordings: true, collection: !!store, releases: releases.map(r => r.name) });
    if (url.pathname === '/api/releases') return replyJson(headers, { releases });
    if (url.pathname === '/manifest.json') return reply(headers, JSON.stringify({ version: 1, collection: 'owner wake-phrase recordings, private/on-host, unreviewed, trainingEligible false',
      resampler: { source: 'eaRS browser SDK audio.ts', sha256: RESAMPLER_SHA }, assets: [...assets.keys()] }, null, 2), 200, 'application/json');
    if (url.pathname === '/speech.json') return reply(headers, JSON.stringify(speech, null, 2), 200, 'application/json');
    // The recorder always gets the owner-reviewed list, so edits change what is recorded next.
    if (url.pathname === '/wordpack.json' && wordpack) return replyJson(headers, lookalikes ? effective(wordpack, lookalikes.read()) : wordpack);
    const asset = assets.get(url.pathname);
    if (!asset || !fs.existsSync(asset.file)) return reply(headers, 'Not found', 404);
    const assetHeaders = asset.csp ? { ...headers, 'Content-Security-Policy': asset.csp } : headers;
    return reply(assetHeaders, request.method === 'HEAD' ? null : Bun.file(asset.file), 200, asset.type);
  };
}

if (import.meta.main) {
  const pinned = await buildResampler();
  const { assets } = verifiedAssets();
  let packs = { assets: new Map(), releases: [] };
  if (process.env.WAQY_PACKS_DIR) {
    const loaded = packsAssets({ packsDir: process.env.WAQY_PACKS_DIR });
    const heads = headsAssets({ webDir: process.env.WAQY_WEB_DIR ?? path.join(HOME, 'byteowlz/waqy/web') });
    for (const [url, asset] of loaded.assets) assets.set(url, asset);
    for (const [url, asset] of heads.assets) assets.set(url, asset);
    packs = loaded;
  }
  let detector = null;
  if (process.env.WAQY_MODELS_DIR) {
    const loaded = detectorAssets({ webDir: process.env.WAQY_WEB_DIR ?? path.join(HOME, 'byteowlz/waqy/web'), modelsDir: process.env.WAQY_MODELS_DIR ?? path.join(HOME, '.local/share/byteowlz/models/waqy/spqx-r3') });
    for (const [url, asset] of loaded.assets) assets.set(url, asset);
    detector = { modelsDir: process.env.WAQY_MODELS_DIR ?? path.join(HOME, '.local/share/byteowlz/models/waqy/spqx-r3'), classifiers: loaded.manifest.classifiers, release: loaded.manifest.release };
  }
  const speech = JSON.parse(fs.readFileSync(path.join(ROOT, 'speech.json'), 'utf8'));
  // WAQY_WORDPACK: the full pack incl. the private roster (names), kept outside git.
  const wordpack = JSON.parse(fs.readFileSync(process.env.WAQY_WORDPACK || path.join(ROOT, 'wordpack.json'), 'utf8'));
  const port = Number(process.env.PORT ?? 8036);
  // RECORDINGS_MAX_COUNT / RECORDINGS_MAX_BYTES move the safety caps (defaults 50k takes / 8 GiB).
  const limits = {};
  for (const [key, env] of [['maxCount', 'RECORDINGS_MAX_COUNT'], ['maxBytes', 'RECORDINGS_MAX_BYTES']]) {
    const value = Number(process.env[env]);
    if (process.env[env] && Number.isFinite(value)) limits[key] = value;
  }
  const store = createRecordingStore({ root: process.env.RECORDINGS_DIR || undefined, ...limits });
  const allowedOrigins = [PUBLIC_ORIGIN, ...EXTRA_ORIGINS, `http://127.0.0.1:${port}`, `http://localhost:${port}`];
  const live = detector ? createLiveStore({ root: process.env.LIVE_DIR || undefined }) : null;
  const lookalikes = createLookalikeStore({ file: process.env.LOOKALIKES_FILE || undefined });
  const handler = createHandler({ assets, speech, store, live, releases: packs.releases, wordpack, lookalikes, allowedOrigins });
  const fetch = process.env.ACCESS_LOG === '1' ? request => { console.log(JSON.stringify({ access: request.method, path: new URL(request.url).pathname })); return handler(request); } : handler;
  const server = Bun.serve({ hostname: '127.0.0.1', port, maxRequestBodySize: 3 * 1024 * 1024, fetch, development: false });
  console.log(JSON.stringify({ service: 'waqy', listen: server.url.href, recordings: store.root, lookalikes: lookalikes.file, live: live?.root ?? null, pinned, detector, releases: packs.releases.map(r => `${r.name}:${r.classifiers.length}`) }));
}

