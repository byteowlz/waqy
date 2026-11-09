// Queue runner integration: real fake-mic capture + server, controlled third-take faults.
// Never point this at the owner's store: RECORDINGS_DIR must be a disposable test directory.
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8043';
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(origin).hostname), 'local test server only');
assert.ok(process.env.RECORDINGS_DIR, 'disposable RECORDINGS_DIR required');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, headless: true,
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-audio-capture=${process.env.AUDIO_FIXTURE}`] });
const checks = [];
async function setup() {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.wakeCollect?.plan().length > 0);
  await page.locator('#record-scope button[data-value="all"]').click();
  await page.locator('#label button[data-value="both"]').click();
  assert.equal(await page.locator('#take-count').inputValue(), '3');
  return { context, page, errors };
}
async function finish({ context, page, errors }) {
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !window.wakeCollect.run() && !window.wakeCollect.active(), null, { timeout: 10000 });
  assert.deepEqual(errors, []);
  await context.close();
}
try {
  const quiet = await setup();
  await quiet.page.evaluate(async () => {
    const { Microphone } = await import('/capture.mjs');
    const slice = Microphone.prototype.interimSlice;
    let calls = 0;
    Microphone.prototype.interimSlice = function (...args) {
      const pcm = slice.apply(this, args);
      // Low gain is not silence: retain this take, flag its level, and advance.
      return ++calls >= 3 ? pcm.map(sample => sample * 0.002) : pcm;
    };
  });
  await quiet.page.locator('#start').click();
  await quiet.page.waitForFunction(() => window.wakeCollect.run()?.saved.length >= 4, null, { timeout: 18000 });
  const saved = await quiet.page.evaluate(() => window.wakeCollect.run().saved);
  const third = JSON.parse(fs.readFileSync(path.join(process.env.RECORDINGS_DIR, saved[2], 'meta.json')));
  assert.ok(third.audio.peakPcm > 0 && third.audio.peakPcm < 32768 * 10 ** (-30 / 20));
  assert.equal(third.trainingEligible, false);
  checks.push('all-mode target=3 advances past a quiet third take; nonzero audio saved unreviewed');
  await finish(quiet);

  const failed = await setup();
  let posts = 0;
  await failed.page.route('**/api/recordings', async route => {
    if (route.request().method() === 'POST' && ++posts === 3)
      await route.fulfill({ status: 507, body: 'Recording store is full' });
    else await route.continue();
  });
  await failed.page.locator('#start').click();
  await failed.page.waitForFunction(() => window.wakeCollect.run()?.paused && !window.wakeCollect.active(), null, { timeout: 18000 });
  assert.match(await failed.page.locator('#status').textContent(), /save|upload/i);
  await failed.page.waitForTimeout(2800);
  assert.equal(posts, 3, 'failed take must not loop in the background');
  assert.equal((await failed.page.evaluate(() => window.wakeCollect.run())).saved.length, 2);
  await failed.page.locator('#start').click();
  await failed.page.waitForFunction(() => window.wakeCollect.run()?.saved.length >= 3, null, { timeout: 10000 });
  checks.push('failed third save pauses with mic closed; explicit Resume retries without losing two saves');
  await finish(failed);

  const stalled = await setup();
  let requests = 0;
  await stalled.page.route('**/api/recordings', async route => {
    // Leave the third upload unanswered to exercise the real fetch deadline.
    if (route.request().method() === 'POST' && ++requests === 3) return;
    await route.continue();
  });
  await stalled.page.locator('#start').click();
  await stalled.page.waitForFunction(() => window.wakeCollect.run()?.paused && !window.wakeCollect.active(), null, { timeout: 25000 });
  assert.match(await stalled.page.locator('#status').textContent(), /could not save/);
  await stalled.page.waitForTimeout(2500);
  const afterTimeout = requests;
  await stalled.page.waitForTimeout(2500);
  assert.equal(requests, afterTimeout, 'an unanswered upload must not leave capture running forever');
  checks.push('unanswered upload reaches the eight-second deadline and pauses without endless retries');
  await finish(stalled);

  const silent = await setup();
  await silent.page.evaluate(async () => {
    const { Microphone } = await import('/capture.mjs');
    const slice = Microphone.prototype.interimSlice;
    let calls = 0;
    Microphone.prototype.interimSlice = function (...args) {
      const pcm = slice.apply(this, args);
      return ++calls >= 3 ? new Float32Array(pcm.length) : pcm;
    };
  });
  await silent.page.locator('#start').click();
  await silent.page.waitForFunction(() => window.wakeCollect.run()?.paused && !window.wakeCollect.active(), null, { timeout: 18000 });
  const stopped = await silent.page.evaluate(() => window.wakeCollect.run());
  assert.equal(stopped.saved.length, 2);
  assert.match(await silent.page.locator('#status').textContent(), /no audio|silent|input/i);
  await silent.page.waitForTimeout(2500);
  assert.equal((await silent.page.evaluate(() => window.wakeCollect.run())).index, stopped.index);
  checks.push('truly silent third take pauses once, leaves saved progress intact, and Stop stays responsive');
  await finish(silent);
  console.log(JSON.stringify({ checks, microphone: 'synthetic source; third-take faults injected, no physical mic test' }, null, 2));
} finally {
  await browser.close();
}
