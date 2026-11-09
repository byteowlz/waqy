// End-to-end check of /detect in headless Chromium with a fake microphone playing a fixture wav.
// Proves: page loads the sha-verified models, onnxruntime-web runs in the worker under the CSP,
// a spoken wake phrase produces a detection, a non-wake fixture does not, and nothing is uploaded.
// Usage: TEST_ORIGIN=http://127.0.0.1:8037 ATTEMPT=detect-browser-N POSITIVE_WAV=... NEGATIVE_WAV=... bun detect-check.mjs
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert';
// playwright-core is a dev dependency of these checks; set PLAYWRIGHT to its module path.
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');

const root = import.meta.dir;
// A Chromium binary and (for the recording check) an audio fixture are external inputs.
const chromiumPath = process.env.CHROMIUM;
const fixture = process.env.AUDIO_FIXTURE;
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8037';
const name = process.env.ATTEMPT ?? 'detect-browser-1';
assert.match(name, /^[a-z0-9-]+$/);
const destination = path.join(process.env.EVIDENCE_DIR ?? path.join(root, 'evidence'), name);
fs.mkdirSync(destination, { recursive: true, mode: 0o700 });
const report = { origin, checks: [], runs: [] };

async function run(label, wav, expectDetection, seconds, mode) {
  const record = mode === 'full';
  const browser = await chromium.launch({ headless: true, executablePath: chromiumPath,
    args: ['--disable-gpu', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-audio-capture=' + wav] });
  const errors = [], posts = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['microphone'], serviceWorkers: 'block' });
    const page = await context.newPage();
    page.on('pageerror', (e) => errors.push(e.message));
    // onnxruntime-web logs this CPU-vendor probe at error level on Apple silicon; it is informational.
    const benign = /cpuid_info\.cc:\d+ LogEarlyWarning\] Unknown CPU vendor/;
    page.on('console', (m) => { if (m.type() === 'error' && !benign.test(m.text())) errors.push(m.text()); });
    page.on('request', (r) => { if (r.method() !== 'GET') posts.push(r.method() + ' ' + r.url()); });
    await page.goto(origin + '/detect', { waitUntil: 'networkidle' });
    await page.locator('#start:not([disabled])').waitFor({ timeout: 30000 });
    await page.locator('#mode').selectOption(mode);
    const started = Date.now();
    await page.locator('#start').click();
    await page.locator('#stop:not([disabled])').waitFor({ timeout: 60000 });
    const loadMs = Date.now() - started;
    const deadline = Date.now() + seconds * 1000;
    let detected = false;
    while (Date.now() < deadline) {
      detected = detected || (await page.locator('#log').textContent()).includes('score');
      if (detected && mode === 'off') break;
      await page.waitForTimeout(250);
    }
    if (record || mode === 'allday') await page.locator('#missed').click();
    let review = null;
    if (mode === 'allday' && detected) {
      await page.locator('.review-row button', { hasText: 'Wrong' }).first().click();
      await page.locator('.review-row .verdict').first().waitFor({ timeout: 10000 });
      review = await page.locator('#tally').textContent();
    }
    const timing = await page.locator('#timing').textContent();
    const log = await page.locator('#log').textContent();
    const scores = await page.$$eval('.score-row', (rows) => rows.map((r) => [r.dataset.id, r.querySelector('.score-last').textContent]));
    await page.screenshot({ path: path.join(destination, `${label}-desktop.png`), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    await page.screenshot({ path: path.join(destination, `${label}-phone.png`), fullPage: true });
    const session = await page.evaluate(() => window.wakeDetect.recorder() ?? window.wakeDetect.day());
    await page.locator('#stop').click();
    await page.waitForFunction(() => document.getElementById('status').textContent.startsWith('Stopped'), { timeout: 20000 });
    let stored = null;
    if (record) {
      const dir = path.join(process.env.LIVE_DIR, session.sessionId);
      const chunks = fs.readFileSync(path.join(dir, 'chunks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
      const scoreRows = fs.readFileSync(path.join(dir, 'scores.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
      const marks = fs.readFileSync(path.join(dir, 'marks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
      const samples = chunks.reduce((a, c) => a + c.samples, 0);
      const contiguous = chunks.every((c, i) => i === 0 ? c.startSample === 0 : c.startSample === chunks[i - 1].startSample + chunks[i - 1].samples);
      stored = { session: session.sessionId.slice(0, 8), chunks: chunks.length, seconds: samples / 16000, contiguous, scoreRows: scoreRows.length,
        maxFrame: Math.max(...scoreRows.map(r => r.f)), detections: scoreRows.filter(r => r.d.length).length, marks: marks.map(m => m.kind) };
      assert.ok(chunks.length >= 2 && contiguous, 'contiguous chunks saved');
      assert.ok(Math.abs(stored.maxFrame * 1280 - samples) < 16000 * 2.5, 'scores cover the recorded audio');
      assert.deepEqual(stored.marks, ['missed']);
      assert.ok(posts.every(p => p.includes('/api/live/')), 'only private live-store POSTs');
    } else if (mode === 'allday') {
      const dir = path.join(process.env.LIVE_DIR, session.sessionId);
      const files = fs.readdirSync(dir).sort();
      const snippets = fs.readFileSync(path.join(dir, 'snippets.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
      const marks = fs.readFileSync(path.join(dir, 'marks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
      const listening = JSON.parse(fs.readFileSync(path.join(dir, 'listening.json'), 'utf8'));
      stored = { files, snippets: snippets.map((x) => [x.kind, +(x.samples / 16000).toFixed(2)]), marks: marks.map((m) => m.kind), listened: listening.listenedSeconds, review };
      assert.ok(!files.some((f) => f.startsWith('chunk-')), 'all-day mode never stores continuous audio');
      assert.ok(snippets.some((x) => x.kind === 'detection') && snippets.some((x) => x.kind === 'missed'), 'detection and missed snippets saved');
      assert.ok(snippets.every((x) => x.samples <= 16000 * 8), 'snippets stay short');
      assert.deepEqual(marks.map((m) => m.kind), ['false-trigger'], 'Wrong button recorded');
      assert.ok(listening.listenedSeconds > 15, 'listening time recorded');
    } else {
      assert.equal(posts.length, 0, 'saving off: no non-GET requests');
    }
    const result = { label, record, wav: path.basename(wav), expectDetection, detected, loadMs, timing, log, scores, errors, posts: posts.length, stored, overflow };
    report.runs.push(result);
    assert.equal(errors.length, 0, 'page errors: ' + errors.join('; '));
    assert.equal(overflow, false, 'no horizontal overflow at 390 px');
    assert.equal(detected, expectDetection, `${label}: detection ${detected}, expected ${expectDetection}`);
    report.checks.push(`${label}: detection=${detected} as expected; ${timing}; ${stored ? `stored ${JSON.stringify(stored)}` : 'nothing saved, no uploads'}; no errors; phone width ok`);
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(destination, 'report.json'), JSON.stringify(report, null, 2));
  }
}

await run('allday', process.env.POSITIVE_WAV, true, 25, 'allday');
await run('full-recording', process.env.POSITIVE_WAV, true, 25, 'full');
if (process.env.NEGATIVE_WAV) await run('nothing-saved', process.env.NEGATIVE_WAV, false, 20, 'off');
console.log(JSON.stringify(report, null, 2));
