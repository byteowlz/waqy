// Look-alike editor end to end against a disposable local server:
// remove / undo / reload persistence / add / homophone refusal / delete stale takes /
// back to recording with the edited queue / phone layout. Never point at the owner's store.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8048';
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(origin).hostname), 'local test server only');
const recordings = process.env.RECORDINGS_DIR, overridesFile = process.env.LOOKALIKES_FILE;
assert.ok(recordings && overridesFile, 'disposable RECORDINGS_DIR and LOOKALIKES_FILE required');
const out = process.env.EVIDENCE_DIR; if (out) fs.mkdirSync(out, { recursive: true });
const checks = [], errors = [];
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, headless: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });

  // Seed one recorded take of a phrase that is no longer a look-alike (a homophone).
  await page.goto(origin + '/lookalikes', { waitUntil: 'networkidle' });
  const seeded = await page.evaluate(async () => {
    const samples = 8000, buffer = new ArrayBuffer(44 + samples * 2), view = new DataView(buffer);
    const text = (o, v) => [...v].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, 'data'); view.setUint32(40, samples * 2, true);
    for (let i = 0; i < samples; i++) view.setInt16(44 + i * 2, Math.round(Math.sin(i / 7) * 4000), true);
    const body = new FormData();
    body.set('audio', new Blob([buffer], { type: 'audio/wav' }), 'take.wav');
    body.set('meta', JSON.stringify({ sessionId: crypto.randomUUID(), language: 'de', group: 'confusable', label: 'negative', take: 1,
      word: 'btw', form: 'confusable', phrase: 'buy the way', spoken: 'buy the way', words: '', style: 'normal' }));
    return (await fetch('/api/recordings', { method: 'POST', body })).json();
  });
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForSelector('.lk-word');
  const state = await page.evaluate(() => window.lookalikeEditor.state());
  assert.equal(await page.locator('.lk-word').count(), state.words.length);
  assert.equal(await page.locator('#rail a.rail-word').count(), state.words.length);
  checks.push(`${state.words.length} words, ${state.counts.active} look-alikes listed with their wake forms`);

  const hexDe = page.locator('#w-hex .lk-lang').filter({ has: page.locator('h3', { hasText: 'German' }) });
  assert.match(await hexDe.locator('.lk-wake').textContent(), /Wake: Hey Hex · He Hex · Hex · Hex\?/);
  const hecks = hexDe.locator('.lk-item[data-key="hex|de|Hecks"]');
  await hecks.getByRole('button', { name: /^Remove/ }).click();
  await page.waitForFunction(() => document.querySelector('[data-key="hex|de|Hecks"]')?.dataset.state === 'removed');
  assert.equal(await page.evaluate(() => document.activeElement?.textContent), 'Restore');
  assert.match(await page.locator('#status').textContent(), /Removed “Hecks” from Hex \(DE\)/);
  await page.locator('#status').getByRole('button', { name: 'Undo' }).click();
  await page.waitForFunction(() => document.querySelector('[data-key="hex|de|Hecks"]')?.dataset.state === 'active');
  checks.push('Remove moves a phrase to Removed with focus on Restore; Undo brings it back');
  await hecks.getByRole('button', { name: /^Remove/ }).click();
  await page.waitForFunction(() => document.querySelector('[data-key="hex|de|Hecks"]')?.dataset.state === 'removed');
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForSelector('[data-key="hex|de|Hecks"]');
  assert.equal(await page.locator('[data-key="hex|de|Hecks"]').getAttribute('data-state'), 'removed');
  assert.deepEqual(JSON.parse(fs.readFileSync(overridesFile, 'utf8')).removed, ['hex|de|Hecks']);
  checks.push('a removal survives reload and is stored in the private overrides file');

  const addField = hexDe.locator('.lk-add input');
  await addField.fill('Häcksler');
  await addField.press('Enter');
  await page.waitForSelector('[data-key="hex|de|Häcksler"]');
  assert.match(await page.locator('[data-key="hex|de|Häcksler"] .lk-meta').textContent(), /added by you/);
  assert.equal(await page.evaluate(() => document.activeElement?.dataset.key), 'add|hex|de');
  await addField.fill('Hex?');
  await addField.press('Enter');
  await page.waitForFunction(() => !document.getElementById('error').hidden);
  assert.match(await page.locator('#error').textContent(), /contains Hex itself/);
  await addField.fill('Hex, die');
  await addField.press('Enter');
  await page.waitForFunction(() => /Hex, die/.test(document.getElementById('error').textContent));
  assert.equal(await page.locator('[data-key="hex|de|Hex, die"]').count(), 0);
  checks.push('Enter adds a look-alike and keeps focus in the field; adding the wake word itself is refused with a reason');

  await page.locator('#search').fill('buy the way');
  const orphan = page.locator('[data-key="btw|de|buy the way"]');
  await orphan.waitFor();
  assert.match(await orphan.locator('.lk-meta').textContent(), /not in the list · 1 take/);
  await orphan.getByRole('button', { name: /^Delete 1 take of/ }).click();
  await orphan.getByRole('button', { name: /^Confirm: delete 1 take of “buy the way”/ }).click();
  await page.waitForFunction(() => !document.querySelector('[data-key="btw|de|buy the way"]'));
  assert.equal(fs.existsSync(path.join(recordings, seeded.id)), false);
  assert.match(await page.locator('#status').textContent(), /Deleted 1 take of “buy the way”/);
  checks.push('a recorded phrase that is no longer a look-alike shows up and its takes delete in two steps');
  await page.locator('#search').fill('');

  // Uniform grid: every action button in the list has one size, and main actions share one x.
  const boxes = await page.$$eval('.lk-item .lk-action, .lk-add .lk-action', nodes => nodes.map(n => { const r = n.getBoundingClientRect(); return { w: Math.round(r.width), h: Math.round(r.height), right: Math.round(r.right), wraps: n.scrollWidth > n.clientWidth }; }));
  assert.equal(new Set(boxes.map(b => `${b.w}x${b.h}`)).size, 1, JSON.stringify([...new Set(boxes.map(b => `${b.w}x${b.h}`))]));
  assert.ok(boxes.every(b => !b.wraps), 'no truncated button labels');
  const mainRight = await page.$$eval('#w-hex .lk-lang:first-of-type .lk-item > :last-child.lk-action, #w-hex .lk-lang:first-of-type .lk-add .lk-action', nodes => [...new Set(nodes.map(n => Math.round(n.getBoundingClientRect().right)))]);
  assert.equal(mainRight.length, 1, JSON.stringify(mainRight));
  checks.push(`all ${boxes.length} action buttons are ${boxes[0].w}x${boxes[0].h} px; main actions and Add share one column`);
  // Desktop keeps German and English side by side (halves the scrolling).
  const columns = await page.$$eval('#w-hex .lk-lang', nodes => new Set(nodes.map(n => Math.round(n.getBoundingClientRect().top))).size);
  assert.equal(columns, 1, 'German and English share one row on desktop');
  await page.evaluate(() => document.getElementById('w-hex').scrollIntoView());
  if (out) await page.screenshot({ path: path.join(out, 'desktop.png'), fullPage: false });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => document.getElementById('w-hex').scrollIntoView());
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.equal(await page.locator('#rail').isVisible(), false);
  const buttonBox = await page.locator('[data-key="hex|de|Hexe"] button').first().boundingBox();
  assert.ok(buttonBox.height >= 40 && buttonBox.x + buttonBox.width <= 390, JSON.stringify(buttonBox));
  if (out) await page.screenshot({ path: path.join(out, 'phone.png'), fullPage: false });
  checks.push('phone: no horizontal scroll, rows keep 36 px controls inside the screen');
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.locator('#back').click();
  await page.waitForFunction(() => window.wakeCollect?.plan().length > 0);
  await page.locator('#rail button[data-word="hex"]').click();
  await page.locator('#language button[data-value="de"]').click();
  await page.locator('#label button[data-value="negative"]').click();
  const lines = await page.evaluate(() => [...new Set(window.wakeCollect.plan().map(item => item.phrase.line))]);
  assert.ok(lines.includes('Häcksler') && !lines.includes('Hecks'), JSON.stringify(lines));
  assert.equal(await page.locator('#nav-lookalikes').getAttribute('href'), '/lookalikes#w-hex');
  checks.push(`back to recording: Hex DE look-alike queue is ${JSON.stringify(lines)}; nav jumps to this word's list`);

  // The refused homophone add is an intended 422; Chromium logs every non-2xx response.
  const expected = /status of 422 \(Unprocessable Entity\)/;
  assert.equal(errors.filter(line => expected.test(line)).length, 2);   // "Hex?" and "Hex, die"
  assert.deepEqual(errors.filter(line => !expected.test(line)), []);
  checks.push('zero page errors (two intended 422s for the refused adds)');
  await context.close();
  console.log(JSON.stringify({ checks, scope: 'synthetic seeded take, isolated stores, Chromium only' }, null, 2));
} finally {
  await browser.close();
}
