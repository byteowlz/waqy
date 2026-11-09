// Recording queue for the owner collection page. Pure functions, no DOM: the queue is always
// "what is still missing", derived from the take store, so a sitting can stop at any moment and
// the next one (any device, after a reload) continues where the store left off.

export const PACES = {
  // first: cue before the first take of a block (mic just opened); cue: "get ready" before each
  // following take; window: speaking time; gap: pause after a take.
  fast: { first: 900, cue: 300, window: 1900, gap: 100 },
  relaxed: { first: 1500, cue: 800, window: 2600, gap: 400 },
};
export const CATEGORIES = { positive: 'positives', negative: 'negatives' };

/** Store key for one phrase; must match the server summary (recordings.mjs byPhrase). */
export const phraseKey = (word, language, label, style, phrase) => `${word}|${language}|${label}|${style}|${phrase}`;

/**
 * Build the take queue.
 *   scope: 'word' (only `word`) | 'all' (every word, protocol order)
 *   category: 'positive' | 'negative' | 'both' (wake phrases first, then look-alikes, per word)
 *   have: { [phraseKey]: savedCount } from the server summary
 * Missing takes only. If one word is complete, a 'word' scope returns a full extra round
 * instead of nothing, so "record more of this word" still works.
 */
export function buildQueue(protocol, { scope, word, language, category, style = 'normal', target = 3, have = {} }) {
  const words = scope === 'all' ? protocol.words : protocol.words.filter(entry => entry.name === word);
  const labels = category === 'both' ? ['positive', 'negative'] : [category];
  const build = missingOnly => {
    const items = [];
    for (const entry of words) {
      for (const label of labels) {
        for (const phrase of entry.languages?.[language]?.[CATEGORIES[label]] ?? []) {
          const saved = have[phraseKey(entry.name, language, label, style, phrase.line)] ?? 0;
          const count = missingOnly ? Math.max(0, target - saved) : target;
          for (let n = 0; n < count; n++) {
            items.push({ word: entry.name, display: entry.display ?? entry.name, language, category: label, style, phrase,
                         take: saved + n + 1, key: phraseKey(entry.name, language, label, style, phrase.line),
                         groupId: label === 'positive' ? phrase.form : 'confusable',
                         groupLabel: label === 'positive' ? phrase.form : 'look-alike' });
          }
        }
      }
    }
    return items;
  };
  const missing = build(true);
  return { items: missing.length || scope === 'all' ? missing : build(false), extra: !missing.length && scope !== 'all' };
}

/** Seconds a queue slice will take at a pace, including one block-start cue per word/category change. */
export function estimateSeconds(items, pace) {
  const p = PACES[pace] ?? PACES.fast;
  let ms = 0, block = null;
  for (const item of items) {
    const next = `${item.word}|${item.category}`;
    ms += (next === block ? p.cue : p.first) + p.window + p.gap;
    block = next;
  }
  return Math.ceil(ms / 1000);
}

/** Microphone blocks: the mic is reopened per word+category, which bounds the PCM buffer. */
export const blockKey = item => `${item.word}|${item.language}|${item.category}`;

/** Remove every queued take of the phrase at `index` (skip phrase); returns the new item list. */
export function skipPhrase(items, index) {
  const key = items[index]?.key;
  if (!key) return items;
  return [...items.slice(0, index), ...items.slice(index).filter(item => item.key !== key)];
}

/** Put a take back at `index` (redo after deleting it). */
export const requeue = (items, index, item) => [...items.slice(0, index), item, ...items.slice(index)];

export const formatDuration = seconds => seconds >= 3600
  ? `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`
  : seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${seconds} s`;
