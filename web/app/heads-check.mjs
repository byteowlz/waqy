// Bounded browser verification: layout fixtures + actual WASM inference with a synthetic fake mic.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8042';
const destination = path.join(process.env.EVIDENCE_DIR ?? '/tmp/waqy-heads-viewport', process.env.ATTEMPT ?? 'browser-1');
fs.mkdirSync(destination, { recursive: true });
const capture = async (page, file) => { if (process.env.CAPTURE !== '0') await page.screenshot({ path: path.join(destination, file) }); };
const report = { checks: [], failures: [], layouts: [], runtime: [], scope: 'Headless Chromium; fake microphone, not a physical phone or OS CPU measurement' };
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM,
  args: ['--disable-gpu', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-audio-capture=' + process.env.POSITIVE_WAV] });
report.browserVersion = browser.version();
async function check(label, action) {
  try { await action(); report.checks.push(label); } catch (error) { report.failures.push({ label, message: String(error.message) }); }
}
async function geometry(page) {
  return page.evaluate(() => {
    const board = document.querySelector('#rows').getBoundingClientRect();
    const rowRects = [...document.querySelectorAll('.score-row')].map(r => r.getBoundingClientRect());
    return { width: innerWidth, height: innerHeight, scrollWidth: document.documentElement.scrollWidth, scrollHeight: document.documentElement.scrollHeight,
      boardBottom: board.bottom, clippedRows: rowRects.filter(r => r.bottom > board.bottom + 1 || r.right > innerWidth + 1 || r.left < -1).length,
      footerBottom: document.querySelector('.footer').getBoundingClientRect().bottom };
  });
}
function fits(g) {
  assert.ok(g.scrollWidth <= g.width + 1 && g.scrollHeight <= g.height + 1, JSON.stringify(g));
  assert.equal(g.clippedRows, 0, JSON.stringify(g));
  assert.ok(g.footerBottom <= g.height + 1, JSON.stringify(g));
}
const uploads = [], errors = [];
async function makePage(width, height) {
  const page = await browser.newPage({ viewport: { width, height }, permissions: ['microphone'], serviceWorkers: 'block' });
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (!['GET', 'HEAD'].includes(request.method())) uploads.push(request.url()); });
  return page;
}
try {
  for (const [label, width, height] of [['desktop',1440,900],['laptop',1280,720],['phone',390,844],['landscape',844,390],['zoom-equivalent',640,450]]) {
    const page = await makePage(width, height);
    try {
      await page.goto(origin + '/heads'); await page.waitForSelector('.score-row');
      await check(`${label}: viewport fits`, async () => { const g = await geometry(page); report.layouts.push({ label, ...g }); fits(g); });
      await capture(page, `${label}-ready.png`);
      await check(`${label}: all heads reachable without scrolling`, async () => {
        const seen = new Set();
        for (let n = 0; n < 30; n++) {
          for (const id of await page.locator('.score-row').evaluateAll(rows => rows.map(r => r.dataset.id))) seen.add(id);
          if (await page.locator('#next').isDisabled()) break;
          await page.click('#next'); fits(await geometry(page));
        }
        assert.equal(seen.size, await page.evaluate(() => window.__heads.headCount));
      });
      await check(`${label}: filter and empty state retain actions`, async () => {
        await page.fill('#filter', 'no-such-word');
        await page.waitForSelector('.empty'); fits(await geometry(page));
        await page.getByRole('button', { name: 'Clear filter' }).click();
        assert.ok(await page.locator('.score-row').count());
      });
      await check(`${label}: keyboard threshold edit and release disclosure`, async () => {
        const field = page.locator('.threshold-input').first(); await field.fill('0.75'); await field.press('Tab');
        assert.equal(await field.inputValue(), '0.75');
        assert.equal(await page.locator('.threshold-marker').first().evaluate(n => n.style.left), '75%');
        await page.locator('#release-options summary').click();
        assert.ok(await page.locator('#releases input').first().isVisible());
        await page.locator('#release-options summary').press('Escape');
        assert.equal(await page.locator('#release-options').evaluate(n => n.open), false);
      });
    } finally { await page.close(); }
  }
  for (const [label, width, height] of [['desktop',1440,900], ['phone',390,844]]) {
    const page = await makePage(width, height);
    try {
      await page.goto(origin + '/heads'); await page.waitForSelector('#releases input', { state: 'attached' });
      await page.evaluate(() => { for (const box of document.querySelectorAll('#releases input')) box.checked = box.value === 'spqx-pack1'; });
      await page.click('#start');
      await check(`${label}: real model peaks and processing stats`, async () => {
        await page.waitForFunction(() => (window.__heads.peak['spqx-pack1::hey_snake'] ?? 0) > .5, null, { timeout: 60000 });
        const data = await page.evaluate(() => window.__heads);
        assert.equal(data.headCount, 24); assert.ok(data.peakInput > .01); assert.ok(data.stats.mean > 0);
        assert.ok(data.modelBytes > 4 * 1024 * 1024); assert.ok(data.stats.realtimePercent >= 0);
        report.runtime.push({ label, ...data }); fits(await geometry(page));
        await capture(page, `${label}-running.png`);
      });
      await check(`${label}: reset while streaming and stop`, async () => {
        await page.click('#reset');
        await page.waitForTimeout(100);
        assert.deepEqual(await page.evaluate(() => window.__heads.peak), {});
        assert.equal(await page.evaluate(() => window.__heads.frames), 0);
        await page.click('#stop'); await page.click('#reset');
        assert.deepEqual(await page.evaluate(() => window.__heads.peak), {});
        assert.equal(await page.evaluate(() => window.__heads.frames), 0);
        assert.equal(await page.locator('#start').isDisabled(), false);
        fits(await geometry(page));
      });
    } finally { await page.close(); }
  }
  const denied = await makePage(390, 844);
  await denied.addInitScript(() => { navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Microphone permission denied', 'NotAllowedError'); }; });
  try {
    await denied.goto(origin + '/heads'); await denied.waitForSelector('.score-row'); await denied.click('#start');
    await check('permission rejection cleans up and fits', async () => {
      await denied.waitForFunction(() => window.__heads.phase === 'error', null, { timeout: 60000 });
      assert.equal(await denied.locator('#start').isDisabled(), false); fits(await geometry(denied));
      await capture(denied, 'phone-permission-error.png');
    });
  } finally { await denied.close(); }
  const late = await makePage(390, 844);
  await late.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = () => new Promise(resolve => {
      window.__lateGrant = async () => { const stream = await original({ audio: true }); window.__lateStream = stream; resolve(stream); };
    });
  });
  try {
    await late.goto(origin + '/heads'); await late.waitForSelector('.score-row'); await late.click('#start');
    await check('stop fences a late microphone permission grant', async () => {
      await late.waitForFunction(() => !!window.__lateGrant, null, { timeout: 60000 });
      await late.click('#stop'); await late.evaluate(() => window.__lateGrant());
      await late.waitForFunction(() => window.__lateStream.getTracks().every(t => t.readyState === 'ended'));
      assert.equal(await late.evaluate(() => window.__heads.phase), 'stopped');
    });
  } finally { await late.close(); }
  await check('no page errors or audio uploads', async () => { assert.deepEqual(errors, []); assert.deepEqual(uploads, []); });
} finally {
  await browser.close();
  fs.writeFileSync(path.join(destination, 'report.json'), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ checks: report.checks.length, failures: report.failures, runtime: report.runtime.map(r => ({ label: r.label, meanMs: r.stats.mean, p95Ms: r.stats.p95, budgetEstimate: r.stats.realtimePercent, modelBytes: r.modelBytes })), evidence: destination }));
if (report.failures.length) process.exitCode = 1;
