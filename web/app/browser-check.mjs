import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert';
// playwright-core is a dev dependency; set PLAYWRIGHT to its module path.
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core');

const root = import.meta.dir;
const chromiumPath = process.env.CHROMIUM;
const fixture = process.env.AUDIO_FIXTURE;
const origin = process.env.TEST_ORIGIN ?? 'http://127.0.0.1:8037';
const recordings = process.env.RECORDINGS_DIR;
const serverLog = process.env.SERVER_LOG;
const name = process.env.ATTEMPT ?? 'collect-browser-1';
assert.match(name, /^[a-z0-9-]+$/);
assert.ok(serverLog && recordings, 'Run against a throwaway local instance with RECORDINGS_DIR and SERVER_LOG');
const destination = path.join(process.env.EVIDENCE_DIR ?? path.join(root, 'evidence'), name);
fs.mkdirSync(destination, { recursive: true, mode: 0o700 });
const requests = [], errors = [], consoleDiagnostics = [], checks = [], recorded = [], missing = [];
let elapsed = null;
const shots = process.env.NO_SHOTS !== '1';
let quietSummary = null;

const browser = await chromium.launch({ headless: true, executablePath: chromiumPath,
  args: ['--disable-gpu', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
         '--use-file-for-fake-audio-capture=' + (fixture ?? ''), '--renderer-process-limit=1'] });
let context;
try {
  context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['microphone'], serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(240000);
  page.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  page.on('response', response => {
    if (response.status() >= 400) missing.push({ status: response.status(), url: response.url() });
    if (response.request().method() !== 'GET') recorded.push({ method: response.request().method(), status: response.status(), path: new URL(response.url()).pathname });
  });
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') consoleDiagnostics.push(message.text()); });

  await page.goto(origin, { waitUntil: 'networkidle' });
  assert.deepEqual(await page.evaluate(() => ({ secure: isSecureContext, isolated: crossOriginIsolated, mic: !!navigator.mediaDevices?.getUserMedia })), { secure: true, isolated: true, mic: true });
  checks.push('trusted HTTPS / cross-origin isolation / microphone API');
  const words = await page.$$eval('#word option', options => options.map(o => o.value));
  assert.ok(words.includes('hex') && words.includes('oqto') && words.includes('ginee'), words.join(','));
  checks.push(`word selector offers ${words.length} wake words including hex, oqto and ginee`);
  // Desktop: one rail row per word replaces both the word dropdown and the old coverage wall.
  assert.equal(await page.locator('#rail button.rail-word').count(), words.length);
  await page.locator('#rail button[data-word="hex"]').click();
  assert.equal(await page.locator('#rail button[data-word="hex"]').getAttribute('aria-current'), 'true');
  assert.equal(await page.locator('#word-title').textContent(), 'Hex');
  assert.equal(await page.locator('#word').isVisible(), false);
  // Regression guard for the cluttered layout: the takes list must start near the recorder,
  // not below a full-page grid.
  const takesTop = await page.locator('#takes').evaluate(node => node.getBoundingClientRect().top + scrollY);
  assert.ok(takesTop < 1100, `takes list starts at ${takesTop}px`);
  checks.push(`word rail lists ${words.length} words; takes list starts at ${Math.round(takesTop)} px on desktop`);
  if (shots) await page.screenshot({ path: path.join(destination, 'desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.equal(await page.locator('#rail').isVisible(), false);
  assert.equal(await page.locator('#word').isVisible(), true);
  checks.push('phone: rail collapses to one word dropdown, no horizontal scroll');
  if (shots) await page.screenshot({ path: path.join(destination, 'phone.png'), fullPage: true });
  await page.setViewportSize({ width: 1280, height: 900 });

  await page.locator('.options summary').click();
  await page.locator('#take-count').selectOption('1');
  await page.locator('#pace').selectOption('relaxed');
  await page.locator('#language button[data-value="de"]').click();
  await page.locator('#label button[data-value="positive"]').click();
  assert.equal(await page.locator('#label button[data-value="positive"]').getAttribute('aria-pressed'), 'true');
  const dePlan = await page.evaluate(() => window.wakeCollect.plan());
  assert.equal(dePlan.length, 4, JSON.stringify(dePlan.map(i => i.phrase.line)));
  assert.deepEqual(dePlan.map(i => i.phrase.line), ['Hey Hex', 'He Hex', 'Hex', 'Hex?']);
  checks.push('German hex plan collects addressed, informal-address, bare and question forms');

  const started = Date.now();
  await page.locator('#start').click();
  await page.waitForFunction(() => !window.wakeCollect.active() && (window.wakeCollect.saved().download?.length ?? 0) >= 4, { timeout: 240000 });
  await page.waitForFunction(() => !window.wakeCollect.run(), { timeout: 30000 }).catch(() => {});
  elapsed = Date.now() - started;
  const saved = await page.evaluate(() => window.wakeCollect.saved().download);
  assert.ok(saved.every(t => t.id) , JSON.stringify(saved));
  assert.ok(elapsed >= 4 * 3800 - 900, `too fast for cue/window budget: ${elapsed}`);
  for (const take of saved) {
    const row = JSON.parse(fs.readFileSync(path.join(recordings, take.id, 'meta.json'), 'utf8'));
    assert.equal(row.word, 'hex');
    assert.equal(row.language, 'de');
    assert.equal(row.label, 'positive');
    assert.equal(row.form, take.phrase.form);
    assert.equal(row.trainingEligible, false);
    assert.equal(row.humanReview, 'pending');
    assert.ok(fs.existsSync(path.join(recordings, take.id, 'audio.wav')));
  }
  checks.push(`one Record press saved ${saved.length} owner hex takes, unreviewed and training-ineligible`);
  assert.equal(await page.locator('#level').isVisible(), true);
  assert.match(await page.locator('#level').textContent(), /^Last take -?\d+ dBFS/);
  checks.push(`per-take level readout shown: "${(await page.locator('#level').textContent()).trim()}"`);
  // Counts come from whole-store aggregates and refresh after every session without a button.
  await page.waitForFunction(() => document.querySelector('#rail button[data-word="hex"] .rail-counts')?.textContent === '4 · 0', null, { timeout: 10000 });
  assert.match(await page.locator('#rail-total').textContent(), /^\d+ takes/);
  assert.equal(await page.locator('#rail .rail-group').first().textContent(), 'Recorded');
  const summary = await (await fetch(`${origin}/api/recordings/summary`)).json();
  assert.equal(summary.byWord.hex.positive, 4);
  checks.push('coverage reflects the new owner takes from whole-store aggregates');

  // Review: play a saved take and delete a different one from the list.
  await page.waitForFunction(() => document.querySelectorAll('#takes .take').length >= 4, null, { timeout: 10000 });
  const reviewRows = await page.locator('#takes .take').count();
  assert.ok(reviewRows >= 4, String(reviewRows));
  await page.locator('#takes .take').first().getByRole('button', { name: /^Play/ }).click();
  await page.waitForFunction(() => !!window.wakeCollect.playing(), null, { timeout: 10000 });
  await page.waitForFunction(() => !window.wakeCollect.playing(), null, { timeout: 15000 });
  checks.push('review list plays a saved take through the private audio route');
  const doomed = await page.evaluate(() => window.wakeCollect.takes()[0].id);
  // No browser dialog: deletion is a two-step in-page confirmation (site-blocked dialogs must not
  // be able to disable pruning).
  await page.locator(`#takes .take[data-id="${doomed}"]`).getByRole('button', { name: 'Delete', exact: true }).click();
  await page.locator(`#takes .take[data-id="${doomed}"]`).getByRole('button', { name: 'Delete for good' }).click();
  await page.waitForFunction(id => !window.wakeCollect.takes().some(t => t.id === id), doomed, { timeout: 10000 });
  assert.equal(fs.existsSync(path.join(recordings, doomed)), false);
  checks.push('review delete removes the WAV and its metadata from the host');
  await page.locator('#filter-look').click();
  assert.equal(await page.locator('#filter-look').getAttribute('aria-pressed'), 'true');
  await page.locator('#filter-all').click();
  checks.push('review filters switch without losing the list');

  await page.locator('#label button[data-value="negative"]').click();
  const negPlan = await page.evaluate(() => window.wakeCollect.plan());
  // Five after the homophone cleanup removed "Hecks" (it is Hex itself, an accepted spelling).
  assert.equal(negPlan.length, 5);
  assert.ok(negPlan.map(i => i.phrase.line).includes('Hexe'));
  checks.push('German look-alike session uses the spec confusables (Hexe, Heck, Hacks …)');
  const beforeNegative = await page.evaluate(() => window.wakeCollect.saved().count);
  await page.locator('#start').click();
  await page.waitForFunction(target => !window.wakeCollect.active() && window.wakeCollect.saved().count >= target,
    beforeNegative + negPlan.length, { timeout: 240000 });
  const negatives = await page.evaluate(() => window.wakeCollect.saved().download.filter(t => t.category === 'negative'));
  assert.equal(negatives.length, negPlan.length);
  for (const take of negatives) {
    const row = JSON.parse(fs.readFileSync(path.join(recordings, take.id, 'meta.json'), 'utf8'));
    assert.equal(row.label, 'negative');
    assert.equal(row.word, 'hex');
    assert.equal(row.form, 'confusable');
    assert.equal(row.mic, 'Default microphone', 'the device label is stored, not a raw device id');
  }
  checks.push(`one Record press saved ${negPlan.length} look-alike negatives with word/form metadata`);

  await page.locator('#language button[data-value="en"]').click();
  await page.locator('#label button[data-value="positive"]').click();
  const enPlan = await page.evaluate(() => window.wakeCollect.plan());
  assert.deepEqual(enPlan.map(i => i.phrase.line), ['Hey Hex', 'Hex', 'Hex?']);
  checks.push('English plan uses the English forms without German-only variants');

  // All-in-one: every word, missing takes only, fast pace; pause / skip / redo while paused / stop;
  // after a reload the same selection resumes from what the store holds.
  await page.locator('#record-scope button[data-value="all"]').click();
  await page.locator('#language button[data-value="de"]').click();
  await page.locator('#label button[data-value="both"]').click();
  await page.locator('#pace').selectOption('fast');
  await page.locator('#take-count').selectOption('1');
  const storeState = () => page.evaluate(async () => {
    const pack = await (await fetch('/wordpack.json')).json();
    const summary = await (await fetch('/api/recordings/summary', { cache: 'no-store' })).json();
    let phrases = 0, recorded = 0;
    for (const w of pack.words) for (const [label, list] of [['positive', 'positives'], ['negative', 'negatives']])
      for (const phrase of w.languages?.de?.[list] ?? []) {
        phrases++;
        if ((summary.byPhrase[`${w.name}|de|${label}|normal|${phrase.line}`] ?? 0) > 0) recorded++;
      }
    return { phrases, recorded, total: summary.total };
  });
  const beforeAll = await storeState();
  const allPlan = await page.evaluate(() => window.wakeCollect.plan());
  assert.equal(allPlan.length, beforeAll.phrases - beforeAll.recorded);
  assert.ok(new Set(allPlan.map(i => i.word)).size > 30, 'queue spans all words');
  assert.ok(!allPlan.some(i => i.word === 'hex' && i.phrase.line === 'Hexe'), 'already recorded phrases are not queued again');
  assert.match(await page.locator('#start').textContent(), /^Record [\d,]+ missing takes · /);
  checks.push(`all-words queue holds only missing takes: ${allPlan.length} of ${beforeAll.phrases} German phrases`);

  const allStarted = Date.now();
  await page.locator('#start').click();
  await page.waitForFunction(() => window.wakeCollect.run()?.saved.length >= 3, null, { timeout: 30000 });
  const perTake = (Date.now() - allStarted) / 3;
  assert.ok(perTake < 3300, `fast pace per take ${perTake} ms`);
  checks.push(`fast pace: ${Math.round(perTake)} ms per take including the block-start cue`);
  assert.equal(await page.locator('#keys').isVisible(), true);
  assert.equal(await page.locator('#skip').getAttribute('aria-label'), 'Skip this phrase');

  await page.keyboard.press('Space');
  await page.waitForFunction(() => window.wakeCollect.run()?.paused && !window.wakeCollect.active(), null, { timeout: 10000 });
  assert.equal(await page.locator('#start').textContent(), 'Resume');
  const pausedAt = await page.evaluate(() => window.wakeCollect.run());
  await page.waitForTimeout(1500);
  assert.equal((await page.evaluate(() => window.wakeCollect.run())).saved.length, pausedAt.saved.length);
  checks.push('Space pauses: microphone closed, nothing recorded while paused');

  await page.keyboard.press('s');
  await page.waitForFunction(key => window.wakeCollect.run()?.current !== key, pausedAt.current, { timeout: 5000 });
  assert.equal((await page.evaluate(() => window.wakeCollect.run())).length, pausedAt.length - 1);
  checks.push('S skips the current phrase for this sitting');

  const redoTarget = pausedAt.saved.at(-1);
  await page.keyboard.press('r');
  await page.waitForFunction(id => !window.wakeCollect.run()?.saved.includes(id), redoTarget, { timeout: 10000 });
  assert.equal(fs.existsSync(path.join(recordings, redoTarget)), false);
  assert.equal((await page.evaluate(() => window.wakeCollect.run())).length, pausedAt.length);
  checks.push('R deletes the last take from the host and queues it again');

  await page.keyboard.press('Space');
  await page.waitForFunction(n => window.wakeCollect.active() && window.wakeCollect.run()?.saved.length >= n, pausedAt.saved.length, { timeout: 20000 });
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !window.wakeCollect.run() && !window.wakeCollect.active(), null, { timeout: 15000 });
  assert.match(await page.locator('#status').textContent(), /takes saved this sitting\..*still missing/);
  checks.push('Esc stops; the status says how much is still missing');

  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.wakeCollect.plan().length > 0, null, { timeout: 10000 });
  const afterAll = await storeState();
  const resumed = await page.evaluate(() => window.wakeCollect.plan());
  assert.equal(await page.locator('#record-scope button[data-value="all"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#pace').inputValue(), 'fast');
  assert.equal(resumed.length, afterAll.phrases - afterAll.recorded);
  assert.equal(resumed.length, allPlan.length - (afterAll.total - beforeAll.total));
  assert.ok(resumed.some(i => i.key === pausedAt.current), 'a skipped phrase comes back in the next sitting');
  checks.push(`reload resumes: same selection and pace, ${resumed.length} missing (${afterAll.total - beforeAll.total} saved this sitting)`);

  if (shots) {
    await page.screenshot({ path: path.join(destination, 'log.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(destination, 'phone-final.png'), fullPage: true });
  }
  const allowed = entry => {
    const url = new URL(entry.url);
    if (url.origin !== origin) return false;
    if (entry.method === 'GET') return true;
    return /^\/api\/recordings(\/[^/]+)?$/.test(url.pathname);
  };
  assert.ok(requests.every(allowed), JSON.stringify(requests.filter(r => !allowed(r))));
  assert.ok(recorded.every(r => /^\/api\/recordings/.test(r.path) && [200, 201].includes(r.status)), JSON.stringify(recorded));
  checks.push('same-origin only: private POST /api/recordings; no uploads or model fetches');
  assert.deepEqual(errors, []);
  assert.deepEqual(missing, [], 'unexpected HTTP failures');
  assert.ok(consoleDiagnostics.every(line => line === '' || /Worklet|source node|getUserMedia/i.test(line)), JSON.stringify(consoleDiagnostics));
  checks.push('zero page errors');

  await Bun.write(path.join(destination, 'summary.json'), JSON.stringify({ origin, checks, elapsedSessionMs: elapsed,
    words, dePlan: dePlan.map(i => i.phrase.line), enPlan: enPlan.map(i => i.phrase.line), negatives: negatives.map(t => t.phrase.line),
    saved: saved.map(t => ({ id: t.id, phrase: t.phrase.line, form: t.phrase.form })), errors, consoleDiagnostics,
    microphone: 'synthetic fake source, not physical capture', physicalPhone: 'not tested' }, null, 2));
  quietSummary = await quietCheck();
} catch (error) {
  await Bun.write(path.join(destination, 'failure.json'), JSON.stringify({ error: error.stack, checks, requests, errors, consoleDiagnostics }, null, 2));
  throw error;
} finally {
  await context?.close();
  await browser.close();
}
if (quietSummary) console.log(JSON.stringify(quietSummary));

// A low-level microphone (separate browser: the fake source is a launch flag): nonzero takes
// must be saved with a review hint, not discarded/retried solely because of their level.
async function quietCheck() {
  const quietFixture = process.env.QUIET_FIXTURE;
  if (!quietFixture) return { quiet: 'skipped: set QUIET_FIXTURE' };
  const quietBrowser = await chromium.launch({ headless: true, executablePath: chromiumPath,
    args: ['--disable-gpu', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
           '--use-file-for-fake-audio-capture=' + quietFixture, '--renderer-process-limit=1'] });
  try {
    const quietContext = await quietBrowser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['microphone'], serviceWorkers: 'block' });
    const quietPage = await quietContext.newPage();
    const posts = [];
    quietPage.on('request', request => { if (request.method() === 'POST') posts.push(request.url()); });
    await quietPage.goto(origin, { waitUntil: 'networkidle' });
    await quietPage.locator('#rail button[data-word="pi"]').click();
    await quietPage.locator('#start').click();
    await quietPage.waitForFunction(() => window.wakeCollect.run()?.saved.length >= 2, null, { timeout: 30000 });
    assert.equal(await quietPage.evaluate(() => window.wakeCollect.active()), true);
    assert.equal(await quietPage.locator('#level').getAttribute('data-quiet'), 'true');
    assert.match(await quietPage.locator('#level').textContent(), /low level; listen back/);
    assert.ok(posts.length >= 2);
    await quietPage.keyboard.press('Escape');
    await quietPage.waitForFunction(() => !window.wakeCollect.run(), null, { timeout: 10000 });
    checks.push('low-level mic: nonzero takes save and advance with a review hint; no level-based auto-pause');
    await Bun.write(path.join(destination, 'summary.json'), JSON.stringify({ ...JSON.parse(fs.readFileSync(path.join(destination, 'summary.json'), 'utf8')), checks }, null, 2));
    return { quiet: 'ok' };
  } finally { await quietBrowser.close(); }
}
