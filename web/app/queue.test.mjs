import { describe, expect, test } from 'bun:test';
import { buildQueue, estimateSeconds, phraseKey, skipPhrase, requeue, blockKey, formatDuration } from './queue.mjs';

const protocol = { words: [
  { name: 'hex', display: 'Hex', languages: {
    de: { positives: [{ line: 'Hey Hex', form: 'addressed' }, { line: 'Hex', form: 'bare' }], negatives: [{ line: 'Hexe', form: 'confusable' }] },
    en: { positives: [{ line: 'Hey Hex', form: 'addressed' }], negatives: [] } } },
  { name: 'pi', display: 'Pi', languages: { de: { positives: [{ line: 'Hey Pi', form: 'addressed' }], negatives: [{ line: 'Pie', form: 'confusable' }] } } },
  { name: 'run', display: 'Run', languages: { en: { positives: [{ line: 'Run', form: 'bare' }], negatives: [] } } },
] };

describe('recording queue', () => {
  test('all words, both categories: wake phrases before look-alikes, per word, protocol order', () => {
    const { items } = buildQueue(protocol, { scope: 'all', language: 'de', category: 'both', target: 1 });
    expect(items.map(i => `${i.word}:${i.category}:${i.phrase.line}`)).toEqual([
      'hex:positive:Hey Hex', 'hex:positive:Hex', 'hex:negative:Hexe', 'pi:positive:Hey Pi', 'pi:negative:Pie']);
  });
  test('only missing takes are queued, numbered after what is already saved (resume across sittings)', () => {
    const have = { [phraseKey('hex', 'de', 'positive', 'normal', 'Hey Hex')]: 2, [phraseKey('hex', 'de', 'positive', 'normal', 'Hex')]: 5 };
    const { items } = buildQueue(protocol, { scope: 'all', language: 'de', category: 'positive', target: 3, have });
    expect(items.map(i => `${i.word}:${i.phrase.line}:${i.take}`)).toEqual(['hex:Hey Hex:3', 'pi:Hey Pi:1', 'pi:Hey Pi:2', 'pi:Hey Pi:3']);
  });
  test('takes in another style do not count toward the normal-voice target', () => {
    const have = { [phraseKey('pi', 'de', 'positive', 'whispered', 'Hey Pi')]: 3 };
    expect(buildQueue(protocol, { scope: 'word', word: 'pi', language: 'de', category: 'positive', target: 3, have }).items).toHaveLength(3);
  });
  test('a complete word offers an extra round; a complete "all" queue is empty', () => {
    const have = { [phraseKey('pi', 'de', 'positive', 'normal', 'Hey Pi')]: 3 };
    const word = buildQueue(protocol, { scope: 'word', word: 'pi', language: 'de', category: 'positive', target: 3, have });
    expect(word.extra).toBe(true);
    expect(word.items.map(i => i.take)).toEqual([4, 5, 6]);
    const full = { ...have, [phraseKey('hex', 'de', 'positive', 'normal', 'Hey Hex')]: 3, [phraseKey('hex', 'de', 'positive', 'normal', 'Hex')]: 3 };
    expect(buildQueue(protocol, { scope: 'all', language: 'de', category: 'positive', target: 3, have: full }).items).toEqual([]);
  });
  test('words without the language are skipped, not errors', () => {
    expect(buildQueue(protocol, { scope: 'all', language: 'en', category: 'both', target: 1 }).items.map(i => i.word)).toEqual(['hex', 'run']);
  });
  test('skip phrase drops the remaining takes of that phrase only; redo puts a take back', () => {
    const { items } = buildQueue(protocol, { scope: 'word', word: 'hex', language: 'de', category: 'positive', target: 2 });
    const skipped = skipPhrase(items, 0);
    expect(skipped.map(i => i.phrase.line)).toEqual(['Hex', 'Hex']);
    expect(requeue(skipped, 1, items[0]).map(i => i.phrase.line)).toEqual(['Hex', 'Hey Hex', 'Hex']);
    expect(skipPhrase(items, 99)).toBe(items);
  });
  test('estimates count one longer cue per word/category block; durations read naturally', () => {
    const { items } = buildQueue(protocol, { scope: 'all', language: 'de', category: 'both', target: 1 });
    // blocks: hex+, hex-, pi+, pi- => 4 first cues (900) + 1 follow-up cue (300); 5 x (1900 + 100)
    expect(estimateSeconds(items, 'fast')).toBe(Math.ceil((4 * 900 + 300 + 5 * 2000) / 1000));
    expect(new Set(items.map(blockKey)).size).toBe(4);
    expect([formatDuration(45), formatDuration(600), formatDuration(4000)]).toEqual(['45 s', '10 min', '1 h 7 min']);
  });
});
