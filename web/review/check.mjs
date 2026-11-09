// Smoke check for both review modes: page loads, assets resolve, and the write endpoints accept
// and reject the right things. Needs a running instance (REVIEW_ORIGIN) with REVIEW_DIR set.
import assert from 'node:assert';

const origin = process.env.REVIEW_ORIGIN ?? 'http://127.0.0.1:8801';
const host = process.env.REVIEW_HOST ?? '127.0.0.1';
const get = (p) => fetch(origin + p, { headers: { Host: host } });
const post = (p, body) => fetch(origin + p, { method: 'POST', headers: { Host: host, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

const clips = await (await get('/clips.json')).json();
assert.ok(Array.isArray(clips) && clips.length, 'clips.json is a non-empty array');
assert.equal((await get('/app.mjs')).status, 200);
assert.equal((await get('/' + clips[0].clip)).status, 200, 'first clip resolves');
const id = clips[0].id;

if (Array.isArray(clips[0].labels)) {
  assert.equal((await post('/api/label', { id, label: 'genuine' })).status, 200, 'label accepted');
  assert.equal((await post('/api/label', { id, label: 'nonsense' })).status, 400, 'unknown label refused');
  assert.equal((await post('/api/transcript', { id, text: 'Okto' })).status, 200, 'transcript accepted');
  assert.equal((await post('/api/transcript', { id, text: '' })).status, 200, 'empty transcript clears the correction');
  assert.equal((await post('/api/transcript', { id: '../etc/passwd', text: 'x' })).status, 400, 'bad id refused');
  const labels = await (await get('/labels.json')).json();
  assert.equal(labels[id].label, 'genuine');
} else {
  const key = `${clips[0].lang}-r${clips[0].round ?? 1}`;
  assert.equal((await post('/api/vote', { lang: key, spelling: clips[0].spelling })).status, 200, 'vote accepted');
  assert.equal((await post('/api/vote', { lang: key, spelling: '' })).status, 400, 'empty vote refused');
}
console.log(JSON.stringify({ ok: true, mode: Array.isArray(clips[0].labels) ? 'labels' : 'rounds', clips: clips.length }));
