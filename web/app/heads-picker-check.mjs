const said = [];
const say = (...args) => { said.push(args.join(' ')); };
// Live test picker on /detect: choose a published pack and any subset of its heads, then run them.
// Proves the page loads only the ticked heads from that pack and the engine scores them live.
const { chromium } = await import(process.env.PLAYWRIGHT);
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, headless: true, args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
const context = await browser.newContext({ permissions: ['microphone'] });
const page = await context.newPage();
const errors = []; page.on('pageerror', e => errors.push(e.message));
await page.goto(process.env.ORIGIN + '/detect', { waitUntil: 'networkidle' });
await page.waitForFunction(() => document.querySelectorAll('#release option').length > 1);
const releases = await page.$$eval('#release option', o => o.map(x => ({ value: x.value, label: x.textContent })));
say('releases', releases.length, releases.slice(0, 3));
const heads = await page.$$eval('#heads input', i => i.map(x => x.value));
say('installed heads', heads);
await page.selectOption('#release', 'spqx-pack1');
await page.waitForFunction(() => document.querySelectorAll('#heads input').length > 3);
const packHeads = await page.$$eval('#heads input', i => i.map(x => x.value));
say('pack1 heads', packHeads.length);
// Pick two heads and load them.
for (const input of await page.$$('#heads input')) { if (await input.isChecked()) await input.uncheck(); }
await page.locator('#heads input[value="hey_hex"]').check();
await page.locator('#heads input[value="hey_elissa"]').check();
await page.click('#apply');
await page.waitForFunction(() => document.querySelector('#rows .score-row[data-id="hey_hex"]'), null, { timeout: 20000 });
say('rows after load', await page.$$eval('#rows .score-row', r => r.map(x => x.dataset.id)));
say('provenance', (await page.textContent('#provenance')).slice(0, 110));
say('errors', errors);
// Mixed selection from the same pack, then the engine must actually run: start and read a score.
await page.locator('#heads input[value="hex"]').check();
await page.click('#apply');
await page.waitForFunction(() => document.querySelectorAll('#rows .score-row').length === 3, null, { timeout: 20000 });
await page.click('#start');
await page.waitForTimeout(4000);
const live = await page.evaluate(() => [...document.querySelectorAll('#rows .score-row')].map(r => ({ id: r.dataset.id, score: r.querySelector('.score-value').textContent })));
say('live scores', JSON.stringify(live));
await page.click('#stop');
say('errors', errors);
await browser.close();
console.log(said.join('\n'));
