// Real WASM with a synthetic fake microphone; no private recordings or physical-device claims.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8062';
const evidence = process.env.EVIDENCE_DIR ?? '/tmp/waqy-head-record-evidence';
fs.mkdirSync(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM,
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'] });
const errors = [], requests = [], checks = [];
async function pageAt(width, height) {
  const page = await browser.newPage({ viewport: { width, height }, permissions: ['microphone'], serviceWorkers: 'block' });
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', r => { if (r.method() === 'POST') requests.push(r.url()); });
  await page.goto(origin + '/heads'); await page.waitForSelector('.score-row');
  return page;
}
function fits(page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1 && document.documentElement.scrollHeight <= innerHeight + 1);
}
try {
  for (const [label, width, height] of [['desktop',1440,900], ['phone',390,844]]) {
    const page = await pageAt(width, height);
    assert.equal(await page.isChecked('#record-audio'), false);
    assert.ok(await fits(page)); checks.push(label + ': default off, viewport fits');
    if (label === 'phone') {
      await page.click('#details summary');
      assert.ok(await page.isVisible('#recording-intent'));
      await page.screenshot({ path: path.join(evidence, 'phone-recording-options.png') });
      await page.close(); continue;
    }
    const before = requests.length;
    await page.click('#start');
    await page.waitForFunction(() => window.__heads.frames > 10, null, { timeout: 90000 });
    await page.click('#stop');
    await page.waitForFunction(() => window.__heads.phase === 'stopped');
    assert.equal(requests.length, before); checks.push('default listening sends no recordings');
    await page.check('#record-audio');
    // Lower thresholds in the synthetic fixture to exercise logging/replay, not to measure accuracy.
    await page.locator('.threshold-input').first().fill('0.001');
    await page.click('#start');
    await page.waitForFunction(() => window.__heads.recording?.saved >= 128000, null, { timeout: 90000 });
    assert.equal(await page.isDisabled('#reset'), true);
    assert.equal(await page.isDisabled('#record-audio'), true);
    const active = await page.evaluate(() => window.__heads);
    assert.equal(active.headCount, 66);
    assert.ok(active.recording.buffered < 128000);
    checks.push('all 66 heads process real WASM; first chunk saved; audio buffer bounded; reset fenced');
    // Changing a view must not affect the recording inventory.
    await page.fill('#filter', 'nothing-matches-this'); await page.click('#stop');
    await page.waitForFunction(() => window.__heads.phase === 'stopped', null, { timeout: 30000 });
    const data = await page.evaluate(() => window.__heads.recording);
    assert.equal(data.saved, data.samples); assert.equal(data.processed, data.samples); assert.equal(data.failed, 0);
    checks.push('stop drains exact captured sample count and saves the final partial chunk');
    await page.click('#details summary');
    await page.waitForSelector('#recorded-crossings option', { state: 'attached' });
    await page.click('#replay-crossing');
    await page.waitForFunction(() => document.querySelector('#recording-player').currentTime > 0);
    await page.click('#mark-crossing'); await page.waitForFunction(() => document.querySelector('#mark-result').textContent.includes('Marked'));
    const download = page.waitForEvent('download'); await page.click('#download-recording');
    const saved = await download; await saved.saveAs(path.join(evidence, 'synthetic-session-index.json'));
    const index = JSON.parse(fs.readFileSync(path.join(evidence, 'synthetic-session-index.json'), 'utf8'));
    assert.equal(index.meta.heads.length, 66); assert.equal(Object.keys(index.meta.modelHashes).length, 66);
    assert.equal(index.meta.intent, 'negative-only');
    assert.equal(index.meta.trainingEligible, false);
    assert.equal(index.marks[0].kind, 'false-trigger');
    assert.ok(index.marks[0].detection.includes('::'));
    assert.equal(index.chunks.reduce((n,c) => n+c.samples,0), data.samples);
    assert.ok(await fits(page));
    await page.screenshot({ path: path.join(evidence, 'desktop-recording-review.png') });
    checks.push('replay, release-qualified mark and downloaded session index work after filter changes');
    await page.close();
  }
  for (const [width, height] of [[1280,720], [844,390], [640,450]]) {
    const page = await pageAt(width, height); assert.ok(await fits(page)); await page.close();
    checks.push(`${width}x${height}: viewport fits with recording control`);
  }
  // Failed saves stop capture and remain retryable without losing the failed audio chunk.
  const fault = await pageAt(390,844);
  await fault.route('**/api/live/chunk', route => route.fulfill({ status: 507, body: 'Full' }));
  await fault.check('#record-audio'); await fault.click('#start');
  await fault.waitForFunction(() => window.__heads.phase === 'error', null, { timeout: 90000 });
  assert.ok((await fault.evaluate(() => window.__heads.recording)).failed > 0);
  await fault.unroute('**/api/live/chunk');
  await fault.click('#details summary'); await fault.click('#retry-recording');
  await fault.waitForFunction(() => window.__heads.recording.failed === 0 && window.__heads.recording.saved === window.__heads.recording.samples);
  assert.ok(await fits(fault)); checks.push('507 stops capture visibly; retry preserves all captured audio');
  await fault.close();
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  fs.writeFileSync(path.join(evidence, 'report.json'), JSON.stringify({ checks, errors, requests }, null, 2));
}
console.log(JSON.stringify({ checks, errors, evidence }));
