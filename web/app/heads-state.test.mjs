import { test, expect } from 'bun:test';
import { HeadBoard, fitPage, formatBytes, phraseLabel, selectedModelBytes } from './heads-state.mjs';
const release = { name: 'test', threshold: .5, classifiers: ['hey_oqto', 'oqto'], head_bytes: { hey_oqto: 100, oqto: 200 }, shared_bytes: 1000 };

test('labels name spoken words, not aliases or implementation identifiers', () => {
  expect(phraseLabel('hey_oqto')).toBe('Hey Oqto');
  expect(phraseLabel('opencode')).toBe('OpenCode');
  expect(phraseLabel('hey_mei-ling')).toBe('Hey Mei-Ling');
  expect(phraseLabel('hey_new')).toBe('Hey Oqto · new');
  expect(typeof phraseLabel('__proto__')).toBe('string');
});

test('pagination fits readable rows without an empty third column', () => {
  const desktop = fitPage(1408, 650, 26);
  expect(desktop.columns).toBe(2);
  expect(desktop.rows * desktop.rowHeight + 24).toBeLessThanOrEqual(650);
  const phone = fitPage(366, 480, 40);
  expect(phone.columns).toBe(1);
  expect(phone.pages).toBeGreaterThan(1);
  expect(phone.rowHeight).toBeGreaterThanOrEqual(44);
});

test('model file bytes count one shared frontend and every loaded classifier; not RAM', () => {
  expect(selectedModelBytes([release])).toBe(1300);
  expect(selectedModelBytes([release, { ...release, name: 'second' }])).toBe(1600);
  expect(selectedModelBytes([{ ...release, head_bytes: {} }])).toBeNull();
  expect(formatBytes(159813)).toBe('156 KiB');
});

test('unknown scores remain unknown; every frame contributes to peaks, recent crossings stay bounded', () => {
  const board = new HeadBoard(); board.setReleases([release]);
  expect(board.heads.get('test::oqto').score).toBeNull();
  const results = Array.from({ length: 30 }, (_, frame) => ({ frame, scores: { 'test::oqto': frame % 2 ? .9 : .1 } }));
  board.observe({ results, samples: 1280 * 30, ms: 150, processedSamples: 1280 * 30 });
  expect(board.frames).toBe(30);
  expect(board.heads.get('test::oqto').peak).toBe(.9);
  expect(board.recent).toHaveLength(8);
  const n = board.recent.length;
  board.observe({ results: [{ frame: 30, scores: { 'test::oqto': .95 } }] });
  expect(board.recent).toHaveLength(n); // sustained high scores do not log every frame
});

test('metrics normalize processing time to audio duration; never claim OS CPU percentage', () => {
  const board = new HeadBoard(); board.setReleases([release]);
  expect(board.stats().mean).toBeNull();
  board.sent = 3840;
  board.observe({ samples: 2560, ms: 8, processedSamples: 2560 });
  expect(board.stats()).toEqual({ mean: 4, p95: 4, realtimePercent: 5, backlogMs: 80, inputDb: null });
  for (let i = 0; i < 200; i++) board.observe({ samples: 1280, ms: 4, processedSamples: 2560 });
  expect(board.timings).toHaveLength(120);
});

test('threshold edits are per head, survive selection changes, and reset clears measurements', () => {
  const board = new HeadBoard(); board.setReleases([release]);
  expect(board.threshold('test::oqto', .7)).toBe(true);
  expect(board.threshold('test::oqto', NaN)).toBe(false);
  expect(board.threshold('test::oqto', 2)).toBe(false);
  expect(board.heads.get('test::hey_oqto').threshold).toBe(.5);
  board.observe({ results: [{ frame: 0, scores: { 'test::oqto': .8 } }] });
  board.reset();
  expect(board.heads.get('test::oqto').peak).toBeNull();
  expect(board.heads.get('test::oqto').threshold).toBe(.7);
  board.setReleases([]); board.setReleases([release]);
  expect(board.heads.get('test::oqto').threshold).toBe(.7);
});
