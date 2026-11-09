import { test, expect, beforeEach } from 'bun:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { applyEdit, editorState, effective, normalize, lookalikeProblem, wakeForms, createLookalikeStore, LookalikeError } from './lookalikes.mjs';
import { createHandler, PUBLIC_ORIGIN, verifiedAssets } from './server.mjs';
import { createRecordingStore } from './recordings.mjs';

const phrase = (line, kind = 'negative', form = 'confusable') => ({ line, spoken: line, kind, form, note: '' });
const base = { version: 2, words: [
  { name: 'hex', display: 'Hex', accept: [], languages: {
    de: { positives: [phrase('Hey Hex', 'positive', 'addressed'), phrase('Hex', 'positive', 'bare'), phrase('Hex?', 'positive', 'bare')],
          negatives: [phrase('Hexe'), phrase('Hecks'), phrase('Text')] },
    en: { positives: [phrase('Hey Hex', 'positive', 'addressed')], negatives: [phrase('hacks')] } } },
  { name: 'jev', display: 'Jev', accept: ['hey', 'okay', 'jev', 'jeff'], languages: {
    de: { positives: [phrase('Hey Jev', 'positive', 'addressed')], negatives: [phrase('Jens')] } } },
] };
const empty = () => ({ version: 1, revision: 0, updatedAt: null, removed: [], added: {} });

test('normalize ignores case, punctuation and spacing', () => {
  expect(normalize('  Hex? ')).toBe('hex');
  expect(normalize('Side-Quest!')).toBe('side quest');
  expect(wakeForms(base.words[0], 'de')).toEqual([['hex']]);
});

test('Yo address variants strip to the word and cannot enter the look-alike list', () => {
  const word = { name: 'hex', accept: ['yo'], languages: { en: { positives: [phrase('Yo Hex', 'positive', 'addressed')], negatives: [] } } };
  expect(wakeForms(word, 'en')).toEqual([['hex']]);
  expect(lookalikeProblem(word, 'en', 'Hex')).toBe('contains');
  expect(lookalikeProblem(word, 'en', 'Yo')).toBe(null);
});

test('remove hides a spec look-alike from the recorder; restore brings it back', () => {
  const removed = applyEdit(base, empty(), { op: 'remove', word: 'hex', language: 'de', line: 'Hecks' });
  expect(removed.revision).toBe(1);
  expect(effective(base, removed).words[0].languages.de.negatives.map(p => p.line)).toEqual(['Hexe', 'Text']);
  // Other language untouched.
  expect(effective(base, removed).words[0].languages.en.negatives.map(p => p.line)).toEqual(['hacks']);
  const restored = applyEdit(base, removed, { op: 'restore', word: 'hex', language: 'de', line: 'Hecks' });
  expect(effective(base, restored).words[0].languages.de.negatives.map(p => p.line)).toEqual(['Hexe', 'Hecks', 'Text']);
});

test('add appends an owner look-alike; removing it deletes it outright', () => {
  const added = applyEdit(base, empty(), { op: 'add', word: 'hex', language: 'de', line: '  Häcksler ' });
  const negatives = effective(base, added).words[0].languages.de.negatives;
  expect(negatives.at(-1)).toEqual({ line: 'Häcksler', spoken: 'Häcksler', kind: 'negative', form: 'confusable', note: '', source: 'owner' });
  const gone = applyEdit(base, added, { op: 'remove', word: 'hex', language: 'de', line: 'Häcksler' });
  expect(gone.added).toEqual({});
  expect(gone.removed).toEqual([]);
});

test('a wake form or accepted spelling is refused as a look-alike: same sound is a positive', () => {
  for (const line of ['Hex', 'hex?', 'HEY HEX']) {
    expect(() => applyEdit(base, empty(), { op: 'add', word: 'hex', language: 'de', line })).toThrow(LookalikeError);
  }
  // German final devoicing: Jeff is listed as an accepted spelling of Jev.
  try { applyEdit(base, empty(), { op: 'add', word: 'jev', language: 'de', line: 'Jeff' }); throw new Error('accepted'); }
  catch (error) { expect(error.status).toBe(422); expect(error.message).toContain('never a look-alike'); }
});

test('a look-alike may not contain the wake word as whole words (the "Elisa, die" bug)', () => {
  const hex = base.words[0], jev = base.words[1];
  expect(lookalikeProblem(hex, 'de', 'Hex, die')).toBe('contains');
  expect(lookalikeProblem(hex, 'de', "Hex's")).toBe('contains');
  expect(lookalikeProblem(hex, 'de', 'die Hex bitte')).toBe('contains');
  // Different words that only share letters are fine.
  expect(lookalikeProblem(hex, 'de', 'Hexe')).toBe(null);
  expect(lookalikeProblem(hex, 'de', 'Hexenschuss')).toBe(null);
  // Address words in accept_stems are screening stems, not homophones.
  expect(lookalikeProblem(jev, 'de', 'hey')).toBe(null);
  expect(lookalikeProblem(jev, 'de', 'Jeff')).toBe('homophone');
  try { applyEdit(base, empty(), { op: 'add', word: 'hex', language: 'de', line: 'Hex, die' }); throw new Error('accepted'); }
  catch (error) { expect(error.status).toBe(422); expect(error.message).toContain('contains Hex itself'); }
});

test('duplicates, removed duplicates, unknown words and bad input are refused', () => {
  const codes = edit => { try { applyEdit(base, removedHecks, edit); return 200; } catch (e) { return e.status; } };
  const removedHecks = applyEdit(base, empty(), { op: 'remove', word: 'hex', language: 'de', line: 'Hecks' });
  expect(codes({ op: 'add', word: 'hex', language: 'de', line: 'hexe' })).toBe(409);
  expect(codes({ op: 'add', word: 'hex', language: 'de', line: 'HECKS' })).toBe(409);
  expect(codes({ op: 'add', word: 'nope', language: 'de', line: 'x' })).toBe(404);
  expect(codes({ op: 'add', word: 'jev', language: 'en', line: 'x' })).toBe(400);
  expect(codes({ op: 'add', word: 'hex', language: 'de', line: '' })).toBe(400);
  expect(codes({ op: 'add', word: 'hex', language: 'de', line: 'x'.repeat(61) })).toBe(400);
  expect(codes({ op: 'remove', word: 'hex', language: 'de', line: 'Hecks' })).toBe(409);
  expect(codes({ op: 'restore', word: 'hex', language: 'de', line: 'Hexe' })).toBe(404);
  expect(codes({ op: 'rename', word: 'hex', language: 'de', line: 'Hexe' })).toBe(400);
});

test('editor state counts takes per look-alike and lists recorded phrases that are no longer listed', () => {
  const overrides = applyEdit(base, empty(), { op: 'remove', word: 'hex', language: 'de', line: 'Hecks' });
  const byPhrase = { 'hex|de|negative|normal|Hecks': 2, 'hex|de|negative|whispered|Hecks': 1, 'hex|de|negative|normal|Sex': 3,
                     'hex|de|positive|normal|Hex': 9, 'hex|de|negative|normal|Hexe': 1 };
  const view = editorState(base, overrides, byPhrase);
  const de = view.words[0].languages.de;
  expect(de.items.find(i => i.line === 'Hecks')).toEqual({ line: 'Hecks', source: 'spec', removed: true, recorded: 3 });
  expect(de.items.find(i => i.line === 'Hexe').recorded).toBe(1);
  expect(de.orphans).toEqual([{ line: 'Sex', recorded: 3 }]);
  expect(de.wake).toEqual(['Hey Hex', 'Hex', 'Hex?']);
  expect(view.counts).toEqual({ active: 4, removed: 1, added: 0, cleanup: 2 });
});

let root, store, lookalikes, handler;
const same = { origin: PUBLIC_ORIGIN, 'sec-fetch-site': 'same-origin', 'content-type': 'application/json' };
const call = (pathname, init = {}) => handler(new Request('http://localhost' + pathname, init));
const post = (edit, headers = same) => call('/api/lookalikes', { method: 'POST', headers, body: JSON.stringify(edit) });
const SESSION = '0f8fad5b-d9cb-469f-a165-70867728950e';
function wav() {
  const samples = 8000, buffer = new ArrayBuffer(44 + samples * 2), view = new DataView(buffer);
  const text = (o, v) => [...v].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i++) view.setInt16(44 + i * 2, Math.round(Math.sin(i / 7) * 4000), true);
  return new Uint8Array(buffer);
}
const take = (phraseLine, label = 'negative') => store.create(wav(), { sessionId: SESSION, language: 'de', group: 'g', label, take: 1,
  word: 'hex', form: label === 'negative' ? 'confusable' : 'bare', phrase: phraseLine, spoken: phraseLine, words: '' });

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'waqy-lookalikes-'));
  store = createRecordingStore({ root: path.join(root, 'recordings'), maxCount: 100 });
  lookalikes = createLookalikeStore({ file: path.join(root, 'private', 'overrides.json') });
  handler = createHandler({ assets: new Map(), speech: {}, store, wordpack: base, lookalikes });
});

test('routes: edits persist, the recorder gets the edited list, stale revisions conflict', async () => {
  const first = await (await call('/api/lookalikes')).json();
  expect(first.revision).toBe(0);
  let response = await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 });
  expect(response.status).toBe(200);
  expect((await response.json()).revision).toBe(1);
  // Persisted privately, outside the served directory.
  expect(JSON.parse(fs.readFileSync(lookalikes.file, 'utf8')).removed).toEqual(['hex|de|Text']);
  expect((fs.statSync(lookalikes.file).mode & 0o777)).toBe(0o600);
  const pack = await (await call('/wordpack.json')).json();
  expect(pack.words[0].languages.de.negatives.map(p => p.line)).toEqual(['Hexe', 'Hecks']);
  expect(pack.lookalikeRevision).toBe(1);
  // A second tab still at revision 0 must not overwrite the first edit.
  response = await post({ op: 'add', word: 'hex', language: 'de', line: 'Häcksler', revision: 0 });
  expect(response.status).toBe(409);
  const conflict = await response.json();
  expect(conflict.state.revision).toBe(1);
  expect(conflict.error).toContain('changed');
  // Homophone refusal reaches the page as a readable message.
  response = await post({ op: 'add', word: 'jev', language: 'de', line: 'Jeff', revision: 1 });
  expect(response.status).toBe(422);
  expect((await response.json()).error).toContain('accepted spelling');
  // A fresh handler (server restart) reads the same file.
  const restarted = createHandler({ assets: new Map(), speech: {}, store, wordpack: base, lookalikes: createLookalikeStore({ file: lookalikes.file }) });
  expect((await (await restarted(new Request('http://localhost/api/lookalikes'))).json()).revision).toBe(1);
});

test('routes: mutations need the page origin; export is a download; body size is bounded', async () => {
  expect((await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 }, { 'content-type': 'application/json' })).status).toBe(403);
  expect((await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 }, { ...same, origin: 'https://evil.example' })).status).toBe(403);
  expect((await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 }, { ...same, 'sec-fetch-site': 'cross-site' })).status).toBe(403);
  expect((await call('/api/lookalikes', { method: 'POST', headers: same, body: 'x'.repeat(3000) })).status).toBe(413);
  expect((await call('/api/lookalikes', { method: 'POST', headers: same, body: '{nope' })).status).toBe(400);
  expect((await call('/api/lookalikes', { method: 'DELETE', headers: same })).status).toBe(405);
  const exported = await call('/api/lookalikes/export');
  expect(exported.headers.get('Content-Disposition')).toContain('attachment');
  expect((await exported.json()).note).toContain('existing heads are unchanged');
  expect(fs.existsSync(lookalikes.file)).toBe(false);   // refused requests never write the file
});

test('routes: takes can be deleted only for removed or unlisted look-alikes, never live ones or wake takes', async () => {
  take('Hexe'); take('Text'); take('Text'); take('Sex'); take('Text', 'positive');
  let response = await post({ op: 'delete-takes', word: 'hex', language: 'de', line: 'Hexe', revision: 0 });
  expect(response.status).toBe(409);   // Hexe is still a live look-alike
  await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 });
  response = await post({ op: 'delete-takes', word: 'hex', language: 'de', line: 'Text', revision: 1 });
  const body = await response.json();
  expect(body.deleted).toBe(2);
  // The positive take with the same text is not a look-alike take and survives.
  expect(store.summary().byPhrase['hex|de|positive|normal|Text']).toBe(1);
  expect(store.summary().byPhrase['hex|de|negative|normal|Text']).toBeUndefined();
  // "Sex" is recorded but no longer in any list: an orphan, so it may be cleaned up.
  expect(body.words[0].languages.de.orphans).toEqual([{ line: 'Sex', recorded: 1 }]);
  expect((await (await post({ op: 'delete-takes', word: 'hex', language: 'de', line: 'Sex', revision: 1 })).json()).deleted).toBe(1);
  expect(store.summary().byPhrase['hex|de|negative|normal|Hexe']).toBe(1);
});

test('a recordings folder removed under a running server does not break the list', async () => {
  fs.rmSync(path.join(root, 'recordings'), { recursive: true });
  const response = await call('/api/lookalikes');
  expect(response.status).toBe(200);
  expect((await response.json()).counts.cleanup).toBe(0);
});

test('an overrides folder removed under a running server is recreated on the next edit', async () => {
  fs.rmSync(path.dirname(lookalikes.file), { recursive: true });
  expect((await post({ op: 'remove', word: 'hex', language: 'de', line: 'Text', revision: 0 })).status).toBe(200);
  expect(fs.existsSync(lookalikes.file)).toBe(true);
});

test('the editor page and its module are served; the server-side module is not', () => {
  const { assets } = verifiedAssets();
  expect(assets.has('/lookalikes')).toBe(true);
  expect(assets.has('/lookalikes-page.mjs')).toBe(true);
  expect(assets.has('/lookalikes.mjs')).toBe(false);
});
