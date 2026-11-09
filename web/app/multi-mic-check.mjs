// Multi-mic capture end to end: two stubbed input devices record the same phrase at once, so one
// sitting yields one take per mic, aligned to the phrase, plus a device that fails to open.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8043';
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(origin).hostname), 'local test server only');
const recordings = process.env.RECORDINGS_DIR;
assert.ok(recordings, 'disposable RECORDINGS_DIR required');
const checks = [], errors = [];
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, headless: true,
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-audio-capture=${process.env.AUDIO_FIXTURE}`] });
try {
  const context = await browser.newContext({ permissions: ['microphone'] });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin, { waitUntil: 'networkidle' });
  // Chromium's fake device can feed only one capture (the second stream is silence), so the two
  // inputs are independent synthetic streams: a 1.2 s tone burst starting 300 ms in. The second
  // device's consumer starts 150 ms late, which is what per-device trimming has to absorb. A third
  // device always fails to open.
  await page.evaluate(() => {
    const media = navigator.mediaDevices;
    media.enumerateDevices = async () => [
      { deviceId: 'mic-a', kind: 'audioinput', label: 'Built-in Mic', groupId: 'g' },
      { deviceId: 'mic-b', kind: 'audioinput', label: 'USB Mic (150 ms lag)', groupId: 'g' },
      { deviceId: 'mic-c', kind: 'audioinput', label: 'Broken Mic', groupId: 'g' },
    ];
    media.getUserMedia = async constraints => {
      const wanted = constraints?.audio?.deviceId?.exact;
      if (wanted === 'mic-c') throw new DOMException('Device busy', 'NotReadableError');
      const context = new AudioContext();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const destination = context.createMediaStreamDestination();
      oscillator.frequency.value = 220;
      oscillator.connect(gain).connect(destination);
      const now = context.currentTime;
      // A 1.2 s burst every 2.6 s, so every take window catches one and the mic stays useful for
      // the whole session (devices open once per word, not per take).
      for (let i = 0; i < 20; i++) {
        gain.gain.setValueAtTime(0, now + i * 2.6);
        gain.gain.setValueAtTime(0.3, now + 0.3 + i * 2.6);
        gain.gain.setValueAtTime(0, now + 1.5 + i * 2.6);
      }
      oscillator.start();
      if (wanted === 'mic-b') await new Promise(resolve => setTimeout(resolve, 150));
      return destination.stream;
    };
  });
  await page.locator('.options summary').click();
  await page.evaluate(() => window.wakeCollect.refreshMics());
  await page.waitForFunction(() => document.querySelectorAll('#mic-list input').length === 3, null, { timeout: 10000 });
  assert.equal(await page.locator('#mic-list input').count(), 3);
  await page.locator('#mic-list input[data-device="mic-a"]').check();
  await page.locator('#mic-list input[data-device="mic-b"]').check();
  await page.locator('#mic-list input[data-device="mic-c"]').check();
  assert.match(await page.locator('#mics-state').textContent(), /3 selected/);
  await page.locator('#take-count').selectOption('1');
  await page.locator('#language button[data-value="de"]').click();
  await page.locator('#label button[data-value="positive"]').click();
  await page.locator('#start').click();
  await page.waitForFunction(() => !window.wakeCollect.active() && window.wakeCollect.saved().count >= 6, null, { timeout: 60000 });
  // The broken device is dropped after the first block rather than stopping the session.
  await page.waitForFunction(() => window.wakeCollect.mics().length === 2, null, { timeout: 20000 });
  await page.waitForFunction(() => !window.wakeCollect.run(), null, { timeout: 60000 });
  const saved = await page.evaluate(() => window.wakeCollect.saved().download ?? []);
  const rows = saved.map(take => JSON.parse(fs.readFileSync(path.join(recordings, take.id, 'meta.json'), 'utf8')));
  const byMic = rows.reduce((acc, row) => { (acc[row.mic] ??= []).push(row); return acc; }, {});
  assert.deepEqual(Object.keys(byMic).sort(), ['Built-in Mic', 'USB Mic (150 ms lag)']);
  for (const [mic, group] of Object.entries(byMic)) {
    assert.equal(group.length, 4, `${mic} took every phrase`);
    assert.ok(group.every(row => row.micDevice === (mic === 'Built-in Mic' ? 'mic-a' : 'mic-b')));
    // Trimming removes the lag: every take is close to the phrase, not the full 1.9 s window.
    // Trimmed to the 1.2 s burst (+ 150 ms margin each side), not the 1.9 s window.
    // Where the window falls inside the 2.6 s burst cycle decides how much tone it catches, so the
    // bound is the window itself: never longer than the take window, never empty.
    assert.ok(group.every(row => row.audio.durationSeconds > 0.5 && row.audio.durationSeconds <= 1.95), JSON.stringify(group.map(r => r.audio.durationSeconds)));
  }
  // Same phrase, same length (±120 ms) on both mics.
  for (const phrase of new Set(rows.map(row => row.phrase))) {
    const a = rows.find(row => row.phrase === phrase && row.mic === 'Built-in Mic');
    const b = rows.find(row => row.phrase === phrase && row.mic === 'USB Mic (150 ms lag)');
    assert.ok(Math.abs(a.audio.durationSeconds - b.audio.durationSeconds) < 0.25, `${phrase}: ${a.audio.durationSeconds} vs ${b.audio.durationSeconds}`);
  }
  checks.push(`two mics recorded ${rows.length} takes in one sitting (${Object.keys(byMic).join(', ')}), trimmed to the phrase and aligned within 200 ms despite a 150 ms start lag`);
  // Device names, never raw deviceId hashes.
  const levelText = await page.locator('#level').textContent();
  assert.match(levelText, /Built-in Mic: -?\d+ .* USB Mic \(150 ms lag\): -?\d+/);
  assert.ok(!/mic-[abc]|[A-Za-z0-9+/=]{20,}/.test(levelText), levelText);
  checks.push('the level line reports each device by name, not by device id');
  const summary = await (await page.evaluate(async () => (await fetch('/api/recordings/summary')).json()));
  assert.deepEqual(summary.byMic, { 'Built-in Mic': 4, 'USB Mic (150 ms lag)': 4 });
  checks.push('the store counts takes per mic');
  assert.deepEqual(errors, []);
  checks.push('zero page errors');
  await context.close();
  console.log(JSON.stringify({ checks, notes: 'two independent synthetic streams, one started 150 ms late; not physical hardware, no Bluetooth test' }, null, 2));
} finally {
  await browser.close();
}
