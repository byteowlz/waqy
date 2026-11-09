// Pronunciation review server: plays labelled clips and records one vote per (language, round).
//
//   REVIEW_DIR=~/.local/share/foxline/waqy-review/<set> PORT=8792 bun server.mjs
//
// REVIEW_DIR holds clips.json ([{id, lang, spelling, text, heard, clip, round?}]) and clips/*.wav.
// Votes land in <REVIEW_DIR>/votes.json, keyed "<lang>-r<round>", so the decision is data and can be
// read back by the training pipeline instead of living in a chat message.
import fs from 'node:fs';
import path from 'node:path';

const SCRIPT = import.meta.dir;                       // page assets (repo)
const ROOT = path.resolve(process.env.REVIEW_DIR ?? import.meta.dir);  // clips + votes (data)
const PORT = Number(process.env.PORT ?? 8792);
const privateConfig = path.resolve(SCRIPT, '../../.internal/review-server.toml');
const settings = fs.existsSync(privateConfig) ? Bun.TOML.parse(fs.readFileSync(privateConfig, 'utf8')) : {};
const HOST = process.env.REVIEW_HOST ?? settings.host ?? '127.0.0.1';
if (typeof HOST !== 'string' || !HOST.trim()) throw new Error('review host must be a nonempty string');
const VOTES = path.join(ROOT, 'votes.json');
const LABELS = path.join(ROOT, 'labels.json');
const TRANSCRIPTS = path.join(ROOT, 'transcripts.json');
const LABEL_KINDS = ['genuine', 'false', 'lookalike', 'unsure'];
const HOSTS = (process.env.REVIEW_ALLOWED_HOSTS ?? `${HOST},127.0.0.1,localhost`).split(',').map((h) => h.trim());
const VOTE_KEY = /^[a-z]{2,3}-r\d+$/;
const headers = (type) => ({ 'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': "default-src 'self'; media-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:" });
const typeOf = (name) => name.endsWith('.html') ? 'text/html' : name.endsWith('.mjs') ? 'text/javascript' : name.endsWith('.wav') ? 'audio/wav' : 'application/json';

const server = Bun.serve({
  hostname: HOST, port: PORT, development: false,
  async fetch(request) {
    const url = new URL(request.url);
    const host = (request.headers.get('host') ?? '').split(':')[0];
    if (!HOSTS.includes(host)) return new Response('Unknown host', { status: 404, headers: headers('text/plain') });
    if (url.pathname === '/api/vote') {
      if (request.method !== 'POST') return new Response('POST only', { status: 405, headers: headers('text/plain') });
      const body = await request.json().catch(() => null);
      if (!body || !VOTE_KEY.test(String(body.lang))) return new Response('bad vote key', { status: 400, headers: headers('text/plain') });
      if (typeof body.spelling !== 'string' || !body.spelling) return new Response('pick a spelling', { status: 400, headers: headers('text/plain') });
      const current = fs.existsSync(VOTES) ? JSON.parse(fs.readFileSync(VOTES, 'utf8')) : {};
      current[body.lang] = body.spelling;
      current[`${body.lang}_at`] = new Date().toISOString();
      fs.writeFileSync(VOTES, JSON.stringify(current, null, 1));
      return new Response(JSON.stringify(current), { headers: headers('application/json') });
    }
    if (url.pathname === '/api/label') {
      if (request.method !== 'POST') return new Response('POST only', { status: 405, headers: headers('text/plain') });
      const body = await request.json().catch(() => null);
      if (!body || typeof body.id !== 'string' || !/^[A-Za-z0-9_.-]{1,80}$/.test(body.id)) {
        return new Response('bad item id', { status: 400, headers: headers('text/plain') });
      }
      if (!LABEL_KINDS.includes(body.label)) return new Response('bad label', { status: 400, headers: headers('text/plain') });
      const current = fs.existsSync(LABELS) ? JSON.parse(fs.readFileSync(LABELS, 'utf8')) : {};
      current[body.id] = { label: body.label, at: new Date().toISOString() };
      fs.writeFileSync(LABELS, JSON.stringify(current, null, 1));
      return new Response(JSON.stringify(current), { headers: headers('application/json') });
    }
    if (url.pathname === '/api/transcript') {
      if (request.method !== 'POST') return new Response('POST only', { status: 405, headers: headers('text/plain') });
      const body = await request.json().catch(() => null);
      if (!body || typeof body.id !== 'string' || !/^[A-Za-z0-9_.-]{1,80}$/.test(body.id)) {
        return new Response('bad item id', { status: 400, headers: headers('text/plain') });
      }
      // Empty text means "the ASR text was right"; a string replaces it. Cap length, strip controls.
      const text = typeof body.text === 'string' ? body.text.replace(/[\u0000-\u001f\u007f]/g, ' ').slice(0, 400).trim() : null;
      if (text === null) return new Response('bad text', { status: 400, headers: headers('text/plain') });
      const current = fs.existsSync(TRANSCRIPTS) ? JSON.parse(fs.readFileSync(TRANSCRIPTS, 'utf8')) : {};
      if (text) current[body.id] = { text, at: new Date().toISOString() };
      else delete current[body.id];
      fs.writeFileSync(TRANSCRIPTS, JSON.stringify(current, null, 1));
      return new Response(JSON.stringify(current), { headers: headers('application/json') });
    }
    if (url.pathname === '/transcripts.json') {
      return new Response(fs.existsSync(TRANSCRIPTS) ? fs.readFileSync(TRANSCRIPTS, 'utf8') : '{}', { headers: headers('application/json') });
    }
    if (url.pathname === '/labels.json') {
      return new Response(fs.existsSync(LABELS) ? fs.readFileSync(LABELS, 'utf8') : '{}', { headers: headers('application/json') });
    }
    if (url.pathname === '/votes.json') {
      return new Response(fs.existsSync(VOTES) ? fs.readFileSync(VOTES, 'utf8') : '{}', { headers: headers('application/json') });
    }
    const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    // page assets come from this directory, clips and votes from REVIEW_DIR
    const candidates = [path.join(SCRIPT, file), path.join(ROOT, file)];
    const target = candidates.find((c) => fs.existsSync(c) && !fs.statSync(c).isDirectory());
    if (!target) return new Response('Not found', { status: 404, headers: headers('text/plain') });
    return new Response(request.method === 'HEAD' ? null : Bun.file(target), { headers: headers(typeOf(file)) });
  },
});
console.log(JSON.stringify({ service: 'waqy-review', root: ROOT, listen: server.url.href }));
