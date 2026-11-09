import fs from 'node:fs';
import path from 'node:path';

// Owner-reviewed look-alike lists. The word specs (training/words/*.json, via wordpack.json)
// stay the base; the owner's removals and additions live in one private overrides file outside
// git. The recorder is served the effective list, so edits change what gets recorded next.
// Existing recordings are never deleted by an edit; deleting them is a separate, explicit step.

export const DEFAULT_FILE = path.join(process.env.HOME ?? '/', '.local/share/foxline/waqy-lookalikes-1/overrides.json');
const LANGS = ['de', 'en'];
const MAX_LINE = 60;
const ADDRESS = /^(hey|he|hei|okay|ok|oye|eh|yo)[\s,]+/;

export class LookalikeError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

/** Spelling key: case, punctuation and spacing do not make two phrases different. */
export const normalize = text => String(text ?? '').normalize('NFKC').toLowerCase()
  .replace(/[\u0000-\u001f\u007f]/g, '').replace(/[.,!?;:"'„“”‚‘’()\u00bf\u00a1-]+/g, ' ').replace(/\s+/g, ' ').trim();
const key = (word, language, line) => `${word}|${language}|${line}`;
const empty = () => ({ version: 1, revision: 0, updatedAt: null, removed: [], added: {} });

const ADDRESS_WORDS = new Set(['hey', 'he', 'hei', 'okay', 'ok', 'oye', 'eh', 'yo']);
const words = text => normalize(text).split(' ').filter(Boolean);

/** This language's wake forms without their address, as word sequences. */
export function wakeForms(entry, language) {
  const out = [];
  for (const phrase of entry.languages?.[language]?.positives ?? []) {
    const bare = words(normalize(phrase.line).replace(ADDRESS, ''));
    if (bare.length && !out.some(form => form.join(' ') === bare.join(' '))) out.push(bare);
  }
  return out;
}

/**
 * Why a phrase cannot be a look-alike, or null. Same rule as training/build_collection.py:
 * it may not contain the wake form as whole words (a streaming detector hears the wake word
 * first), and it may not be an accepted spelling. accept_stems are language-agnostic ASR
 * screening stems, so address words in them (hey, okay) do not count.
 */
export function lookalikeProblem(entry, language, text) {
  const tokens = words(text);
  for (const form of wakeForms(entry, language)) {
    for (let i = 0; i + form.length <= tokens.length; i++) {
      if (form.every((word, j) => tokens[i + j] === word)) return 'contains';
    }
  }
  const stems = (entry.accept ?? []).map(stem => words(stem).join(' ')).filter(stem => stem && !ADDRESS_WORDS.has(stem));
  return stems.includes(tokens.join(' ')) ? 'homophone' : null;
}
const problemMessage = (problem, text, name) => problem === 'contains'
  ? `“${text}” contains ${name} itself. A look-alike must not include the wake word, or the detector learns to ignore it.`
  : `“${text}” is an accepted spelling of ${name}. Same sound means it is a wake form, never a look-alike.`;

/** The base word pack with the owner's removals and additions applied. */
export function effective(base, overrides) {
  const removed = new Set(overrides.removed);
  const words = base.words.map(entry => {
    const languages = {};
    for (const [language, lists] of Object.entries(entry.languages)) {
      const negatives = lists.negatives
        .filter(phrase => !removed.has(key(entry.name, language, phrase.line)))
        .map(phrase => ({ ...phrase, source: 'spec' }));
      for (const line of overrides.added[`${entry.name}|${language}`] ?? []) {
        negatives.push({ line, spoken: line, kind: 'negative', form: 'confusable', note: '', source: 'owner' });
      }
      languages[language] = { ...lists, negatives };
    }
    return { ...entry, languages };
  });
  return { ...base, words, lookalikeRevision: overrides.revision };
}

/**
 * Editor view: every look-alike per word and language, its source, whether the owner removed
 * it, how many takes are recorded, and recorded phrases no longer in any list (orphans).
 */
export function editorState(base, overrides, byPhrase = {}) {
  const removed = new Set(overrides.removed);
  const recorded = new Map();
  for (const [phraseKey, count] of Object.entries(byPhrase)) {
    const [word, language, label, , line] = phraseKey.split('|');
    if (label !== 'negative') continue;
    const k = key(word, language, line);
    recorded.set(k, (recorded.get(k) ?? 0) + count);
  }
  let active = 0, removedCount = 0, addedCount = 0, cleanup = 0;
  const words = base.words.map(entry => {
    const languages = {};
    for (const language of LANGS) {
      const lists = entry.languages[language];
      if (!lists) continue;
      const listed = new Set();
      const items = [];
      for (const phrase of lists.negatives) {
        const k = key(entry.name, language, phrase.line);
        listed.add(k);
        const isRemoved = removed.has(k);
        isRemoved ? removedCount++ : active++;
        if (isRemoved && recorded.get(k)) cleanup++;
        items.push({ line: phrase.line, source: 'spec', removed: isRemoved, recorded: recorded.get(k) ?? 0 });
      }
      for (const line of overrides.added[`${entry.name}|${language}`] ?? []) {
        const k = key(entry.name, language, line);
        listed.add(k);
        active++; addedCount++;
        items.push({ line, source: 'owner', removed: false, recorded: recorded.get(k) ?? 0 });
      }
      const orphans = [];
      for (const [k, count] of recorded) {
        const [word, lang, line] = k.split('|');
        if (word === entry.name && lang === language && !listed.has(k)) { orphans.push({ line, recorded: count }); cleanup++; }
      }
      languages[language] = { wake: lists.positives.map(phrase => phrase.line), items, orphans };
    }
    return { name: entry.name, display: entry.display, accept: entry.accept ?? [], languages };
  });
  return { revision: overrides.revision, updatedAt: overrides.updatedAt, counts: { active, removed: removedCount, added: addedCount, cleanup }, words };
}

const cleanLine = line => (typeof line === 'string' ? line.normalize('NFKC').replace(/[\u0000-\u001f\u007f]/g, '').replace(/\s+/g, ' ').trim() : '');

function removeEdit(ctx, next) {
  if (ctx.spec) {
    if (next.removed.includes(ctx.k)) throw new LookalikeError(409, 'Already removed');
    next.removed.push(ctx.k);
    return;
  }
  if (!ctx.added.includes(ctx.text)) throw new LookalikeError(404, 'Not in this list');
  next.added[ctx.listKey] = ctx.added.filter(item => item !== ctx.text);
  if (!next.added[ctx.listKey].length) delete next.added[ctx.listKey];
}
function restoreEdit(ctx, next) {
  if (!ctx.spec || !next.removed.includes(ctx.k)) throw new LookalikeError(404, 'Nothing to restore');
  const problem = lookalikeProblem(ctx.entry, ctx.language, ctx.text);
  if (problem) throw new LookalikeError(422, problemMessage(problem, ctx.text, ctx.name));
  next.removed = next.removed.filter(item => item !== ctx.k);
}
function addEdit(ctx, next) {
  const norm = normalize(ctx.text);
  if (!norm) throw new LookalikeError(400, 'A look-alike needs letters');
  const problem = lookalikeProblem(ctx.entry, ctx.language, ctx.text);
  if (problem) throw new LookalikeError(422, problemMessage(problem, ctx.text, ctx.name));
  const existing = [...ctx.entry.languages[ctx.language].negatives.map(phrase => phrase.line), ...ctx.added];
  const duplicate = existing.find(item => normalize(item) === norm);
  if (duplicate && ctx.spec && next.removed.includes(ctx.k)) throw new LookalikeError(409, `“${duplicate}” is in the list but removed; use Restore.`);
  if (duplicate) throw new LookalikeError(409, `“${duplicate}” is already in the list.`);
  next.added[ctx.listKey] = [...ctx.added, ctx.text];
}
const EDITS = { remove: removeEdit, restore: restoreEdit, add: addEdit };

/** Validate one edit against the base pack; returns the new overrides (pure, no I/O). */
export function applyEdit(base, overrides, { op, word, language, line }) {
  const entry = base.words.find(candidate => candidate.name === word);
  if (!entry) throw new LookalikeError(404, 'Unknown wake word');
  if (!LANGS.includes(language) || !entry.languages[language]) throw new LookalikeError(400, 'Unknown language for this word');
  const text = cleanLine(line);
  if (!text || text.length > MAX_LINE) throw new LookalikeError(400, `A look-alike needs 1 to ${MAX_LINE} characters`);
  const apply = EDITS[op];
  if (!apply) throw new LookalikeError(400, 'Unknown edit');
  const listKey = `${word}|${language}`;
  const ctx = { entry, language, text, listKey, name: entry.display ?? word, k: key(word, language, text),
                spec: entry.languages[language].negatives.find(phrase => phrase.line === text), added: overrides.added[listKey] ?? [] };
  const next = { ...overrides, removed: [...overrides.removed], added: { ...overrides.added } };
  apply(ctx, next);
  next.revision = overrides.revision + 1;
  next.updatedAt = new Date().toISOString();
  return next;
}

/** Private overrides file: atomic writes, optimistic revision check. */
export function createLookalikeStore({ file = DEFAULT_FILE } = {}) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const read = () => {
    if (!fs.existsSync(file)) return empty();
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (data.version !== 1 || !Number.isInteger(data.revision) || !Array.isArray(data.removed) || typeof data.added !== 'object') {
      throw new Error(`Unreadable look-alike overrides: ${file}`);
    }
    return data;
  };
  return {
    file,
    read,
    edit(base, edit, expectedRevision) {
      const current = read();
      if (expectedRevision !== current.revision) throw new LookalikeError(409, 'The list changed in another tab or device; reloaded.');
      const next = applyEdit(base, current, edit);
      fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });   // folder may be removed under a running server
      const temp = `${file}.${process.pid}.tmp`;
      fs.writeFileSync(temp, JSON.stringify(next, null, 1) + '\n', { mode: 0o600 });
      fs.renameSync(temp, file);
      return next;
    },
  };
}
