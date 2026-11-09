import { HeadBoard, fitPage, formatBytes, selectedModelBytes } from '/heads-state.mjs';
import { HeadMicrophone } from '/heads-capture.mjs';
import { HeadRecording } from '/heads-recording.mjs';
import { HeadReview } from '/heads-review.mjs';
import { wavBlob } from '/capture.mjs';

const $ = id => document.getElementById(id);
const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
const board = new HeadBoard();
let releases = [], selected = [], phase = 'inventory', page = 0, layout = { pages: 1, pageSize: 1, columns: 1 };
let rows = new Map(), worker = null, mic = null, run = 0, epoch = 0, cancelLoad = null, loadMs = null, lastRecent = '';
let recording = null, stopping = null, drained = null;
const busy = () => ['loading', 'permission', 'warming', 'listening', 'saving'].includes(phase);
const review = new HeadReview(busy, error => { $('recording-problem').hidden = false; $('recording-problem').textContent = String(error?.message ?? error); });

// Read-only diagnostics used by browser checks. No audio or callable test hook is exposed.
Object.defineProperty(window, '__heads', { get: () => ({
  phase, frames: board.frames, peakInput: board.maxInputPeak, headCount: board.heads.size,
  peak: Object.fromEntries([...board.heads].filter(([, h]) => h.peak !== null).map(([key, h]) => [key, h.peak])),
  recording: recording && { sessionId: recording.sessionId, samples: recording.samples, saved: recording.saved, processed: recording.processed, events: recording.events.length, failed: recording.failed.length, buffered: recording.parts.reduce((n, p) => n + p.length, 0) },
  stats: board.stats(), modelBytes: selectedModelBytes(selected), page, pageSize: layout.pageSize, pages: layout.pages,
}) });

function setPhase(next, text) {
  const focused = document.activeElement;
  phase = next; document.body.dataset.phase = next; $('status').textContent = text;
  $('start').disabled = busy() || !board.heads.size;
  $('stop').disabled = !busy() || next === 'saving';
  $('start').hidden = busy(); $('stop').hidden = !busy();
  $('stop').textContent = ['loading', 'permission'].includes(next) ? 'Cancel' : 'Stop';
  if (focused === $('start') && busy()) $('stop').focus({ preventScroll: true });
  if (focused === $('stop') && !busy()) $('start').focus({ preventScroll: true });
  $('reset').disabled = !board.heads.size || ['loading', 'permission', 'saving'].includes(next) || (busy() && !!recording);
  $('releases').disabled = busy();
  $('release-hint').textContent = busy() ? 'Stop before changing the selected releases.' : 'Choose versions to compare. Threshold edits stay on this page.';
  review.render(recording);
}
async function finishStop(drain) {
  const capturePhase = phase;
  cancelLoad?.(); cancelLoad = null; mic?.stop(); mic = null;
  setPhase('saving', recording ? 'Finishing scores and saving audio…' : 'Stopping…');
  if (recording && worker && drain && ['warming', 'listening'].includes(capturePhase)) {
    let timeout;
    const complete = await new Promise(resolve => {
      drained = () => resolve(true); timeout = setTimeout(() => resolve(false), 5000);
      worker.postMessage({ type: 'drain', epoch });
    });
    clearTimeout(timeout); drained = null;
    if (!complete) recording.error = 'Scores did not finish. Audio is retained, but the processed-sample watermark marks the unscored tail.';
  }
  run++; worker?.terminate(); worker = null;
  if (recording) await recording.finish();
  board.sent = board.processed;
  const result = recording?.error ? 'Stopped · recording needs attention in Details' : recording ? 'Stopped · recording saved for review' : 'Stopped · last run';
  setPhase('stopped', result); renderStats();
}
function stop(drain = true) {
  if (!stopping) stopping = finishStop(drain).finally(() => { stopping = null; });
  return stopping;
}
async function fail(error) {
  await stop(false);
  const message = String(error?.message ?? error).slice(0, 2048);
  $('error').textContent = message.length > 160 ? `${message.slice(0, 160)}… Open Details.` : message;
  $('error').hidden = false; $('error-detail').textContent = message; $('error-detail').hidden = false;
  setPhase('error', 'Could not run · check the message above');
}
function selectReleases() {
  const names = new Set([...$('releases').querySelectorAll('input:checked')].map(box => box.value));
  selected = releases.filter(release => names.has(release.name));
  board.setReleases(selected); page = 0; loadMs = null;
  $('release-count').textContent = `· ${selected.length}`;
  $('provenance').textContent = selected.length ? selected.map(r => `${r.name}: ${r.classifiers.length} heads`).join(' · ') : 'No release selected.';
  setPhase('idle', selected.length ? 'Ready · microphone off' : 'Choose a release');
  renderPage(); renderStats();
}
function renderReleases() {
  for (const release of releases) {
    const label = el('label', undefined, 'release-choice');
    const box = el('input'); box.type = 'checkbox'; box.value = release.name; box.checked = true;
    box.onchange = selectReleases;
    const name = el('span'); name.append(el('strong', release.name), el('small', `${release.classifiers.length} heads · default threshold ${Number(release.threshold).toFixed(2)}`));
    label.append(box, name); $('releases').append(label);
  }
}
function createRow(head) {
  const row = el('div', undefined, 'score-row'); row.dataset.id = head.key;
  const word = el('div', undefined, 'word'); word.title = `${head.label} · ${head.release} · ${formatBytes(head.bytes)}`;
  const meta = el('small');
  meta.append(el('span', head.release.replace(/^spqx-/, ''), 'release-code'), el('span', formatBytes(head.bytes), 'model-bytes'));
  word.append(el('strong', head.label), meta);
  const meter = el('div', undefined, 'meter'); meter.setAttribute('role', 'meter');
  meter.setAttribute('aria-label', `Score for ${head.label}, ${head.release}`); meter.setAttribute('aria-valuemin', '0'); meter.setAttribute('aria-valuemax', '1');
  const fill = el('span', undefined, 'fill'), peak = el('span', undefined, 'peak-marker'), mark = el('span', undefined, 'threshold-marker');
  meter.append(fill, peak, mark);
  const value = el('output', '—', 'live-value');
  const input = el('input', undefined, 'threshold-input');
  input.type = 'number'; input.min = '0'; input.max = '1'; input.step = '0.05'; input.value = head.threshold.toFixed(2);
  input.setAttribute('aria-label', `Threshold for ${head.label}, ${head.release}`); input.title = 'Threshold · this page only';
  input.oninput = () => {
    const valid = board.threshold(head.key, input.valueAsNumber);
    input.setAttribute('aria-invalid', String(!valid));
    if (valid) { recording?.threshold(head.key, head.threshold); renderScores(); }
  };
  input.onchange = () => { input.value = head.threshold.toFixed(2); input.removeAttribute('aria-invalid'); };
  row.append(word, meter, value, input);
  rows.set(head.key, { row, meter, fill, peak, mark, value, input });
  return row;
}
function renderPage() {
  const activeKey = document.activeElement?.closest('.score-row')?.dataset.id;
  const query = $('filter').value.trim().toLowerCase();
  const baseWord = head => head.label.replace(/^Hey /, '').split(' · ')[0];
  const list = [...board.heads.values()].filter(h => `${h.label} ${h.release}`.toLowerCase().includes(query))
    .sort((a, b) => baseWord(a).localeCompare(baseWord(b), 'en') || a.release.localeCompare(b.release, 'en') || Number(a.label.startsWith('Hey ')) - Number(b.label.startsWith('Hey ')) || a.label.localeCompare(b.label, 'en'));
  const root = $('rows');
  layout = fitPage(root.clientWidth, root.clientHeight, list.length, parseFloat(getComputedStyle(document.documentElement).fontSize));
  page = Math.min(page, layout.pages - 1);
  const visible = list.slice(page * layout.pageSize, (page + 1) * layout.pageSize);
  const columns = Math.max(1, Math.min(layout.columns, Math.ceil(visible.length / layout.rows)));
  root.style.setProperty('--columns', String(columns)); root.style.setProperty('--row-height', `${layout.rowHeight}px`);
  root.replaceChildren(); rows = new Map();
  if (!visible.length) {
    const empty = el('div', undefined, 'empty');
    empty.append(el('h2', phase === 'inventory' ? 'Loading wake words' : query ? 'No matching wake words' : 'Choose your releases'));
    empty.append(el('p', query ? 'The filter changes this view only. Clear it to see all selected heads.' : 'Open Releases to choose the models to compare. Nothing records until you press Start.'));
    const action = el('button', query ? 'Clear filter' : 'Choose releases');
    action.onclick = () => { if (query) { $('filter').value = ''; renderPage(); } else { $('release-options').open = true; } };
    empty.append(action); root.append(empty);
  } else {
    const perColumn = Math.ceil(visible.length / columns);
    for (let c = 0; c < columns; c++) {
      const column = el('div', undefined, 'score-column'), heading = el('div', undefined, 'column-head');
      for (const label of ['Wake word', 'Signal', 'Score', 'Threshold']) heading.append(el('span', label));
      column.append(heading);
      for (const head of visible.slice(c * perColumn, (c + 1) * perColumn)) column.append(createRow(head));
      root.append(column);
    }
  }
  $('page-label').textContent = list.length ? `${page * layout.pageSize + 1}–${page * layout.pageSize + visible.length} / ${list.length}` : '0 words';
  $('prev').disabled = page === 0; $('next').disabled = page === layout.pages - 1;
  renderScores();
  if (activeKey && rows.has(activeKey)) rows.get(activeKey).input.focus({ preventScroll: true });
}
function renderScores() {
  for (const [key, nodes] of rows) {
    const head = board.heads.get(key);
    nodes.value.textContent = head.score === null ? '—' : head.score.toFixed(2);
    nodes.fill.style.width = `${(head.score ?? 0) * 100}%`;
    nodes.peak.hidden = head.peak === null;
    nodes.peak.style.left = `${(head.peak ?? 0) * 100}%`;
    nodes.mark.style.left = `${head.threshold * 100}%`;
    nodes.meter.title = `Live ${head.score?.toFixed(2) ?? 'unmeasured'} · peak ${head.peak?.toFixed(2) ?? 'unmeasured'} · threshold ${head.threshold.toFixed(2)}`;
    if (head.score === null) {
      for (const name of ['role', 'aria-valuenow', 'aria-valuemin', 'aria-valuemax']) nodes.meter.removeAttribute(name);
    } else {
      nodes.meter.setAttribute('role', 'meter'); nodes.meter.setAttribute('aria-valuemin', '0'); nodes.meter.setAttribute('aria-valuemax', '1');
      nodes.meter.setAttribute('aria-valuenow', String(head.score));
    }
    nodes.row.dataset.above = String(head.above);
  }
  const recent = board.recent[0];
  const fingerprint = recent ? `${recent.key}:${recent.time}:${recent.score}` : '';
  if (fingerprint !== lastRecent) {
    lastRecent = fingerprint; $('recent').replaceChildren();
    for (const r of board.recent) $('recent').append(el('li', `${r.time.toFixed(1)} s · ${r.label} · ${r.release} · ${r.score.toFixed(2)}`));
    if (!recent) $('recent').append(el('li', 'No crossings yet.'));
  }
  $('last-crossing').textContent = recent ? ` · ${recent.label} ${recent.score.toFixed(2)}` : '';
}
function renderStats() {
  const stats = board.stats();
  const ms = value => value === null ? '—' : `${value.toFixed(1)} ms`;
  $('head-count').textContent = `· ${board.heads.size} heads`;
  const bytes = selectedModelBytes(selected);
  $('model-size').textContent = bytes === null ? 'Unknown' : formatBytes(bytes);
  $('mean-ms').textContent = ms(stats.mean); $('p95-ms').textContent = ms(stats.p95);
  $('rt-budget').textContent = stats.realtimePercent === null ? '—' : `${stats.realtimePercent.toFixed(1)}%`;
  $('backlog').textContent = busy() ? `${Math.round(stats.backlogMs)} ms` : '—';
  $('input-level').textContent = busy() && stats.inputDb !== null ? `${stats.inputDb.toFixed(0)} dBFS` : '—';
  $('runtime-detail').textContent = `${board.frames} scored frames · one WASM CPU worker · ${loadMs === null ? 'models not loaded' : `model load ${(loadMs / 1000).toFixed(2)} s`} · shared frontend counted once. File sizes exclude the browser runtime and are not RAM.`;
}
function feedMicrophone(token, pcm) {
  if (token !== run) return;
  try { recording?.push(pcm); } catch (error) { fail(error); return; }
  board.inputPeak = pcm.reduce((peak, sample) => Math.max(peak, Math.abs(sample)), 0);
  board.maxInputPeak = Math.max(board.maxInputPeak, board.inputPeak);
  board.sent += pcm.length;
  if (board.sent - board.processed > 48000) { fail('Processing cannot keep up. Stop and choose fewer releases.'); return; }
  worker?.postMessage({ type: 'pcm', pcm, epoch }, [pcm.buffer]);
}
async function start() {
  selectReleases();
  if (!board.heads.size) return;
  $('error').hidden = true; $('error-detail').hidden = true; $('release-options').open = false; $('details').open = false;
  review.render(null); recording = null;
  if ($('record-audio').checked) recording = new HeadRecording({ heads: [...board.heads.keys()], wav: wavBlob,
    intent: $('recording-intent').value, onChange: () => review.render(recording) });
  const token = ++run; epoch = 0; setPhase('loading', 'Loading models…');
  const started = performance.now();
  worker = new Worker('/heads-worker.mjs', { type: 'module' });
  let timer;
  try {
    await new Promise((resolve, reject) => {
      cancelLoad = () => reject(new Error('Cancelled'));
      timer = setTimeout(() => reject(new Error('Model loading timed out. Try fewer releases.')), 60000);
      worker.onmessage = ({ data }) => {
        if (token !== run) return;
        if (data.type === 'ready') {
          if (data.heads.length !== board.heads.size || data.heads.some(h => !board.heads.has(h))) { reject(new Error('Release inventory changed. Reload this page.')); return; }
          if (recording) recording.modelHashes = data.modelHashes;
          resolve();
        } else if (data.type === 'frame' && data.epoch === epoch) {
          recording?.observe(data, board.heads);
          board.observe(data); renderScores(); renderStats();
          if (board.frames % 13 === 0) review.render(recording);
          if (phase === 'warming' && board.frames) setPhase('listening', `Listening · ${board.heads.size} heads`);
        } else if (data.type === 'drained' && data.epoch === epoch) { drained?.(); }
        else if (data.type === 'error') { reject(new Error(data.message)); fail(data.message); }
      };
      worker.onerror = event => { reject(new Error(event.message)); if (token === run) fail(event.message); };
      worker.postMessage({ type: 'load', releases: selected.map(r => r.name), epoch });
    });
    clearTimeout(timer); cancelLoad = null;
    if (token !== run) return;
    loadMs = performance.now() - started; setPhase('permission', 'Waiting for microphone…');
    mic = new HeadMicrophone();
    const active = await mic.start(pcm => feedMicrophone(token, pcm));
    if (active && token === run) { setPhase('warming', 'Warming up…'); renderStats(); }
  } catch (error) { if (token === run) fail(error); }
  finally { clearTimeout(timer); }
}
function reset() {
  if (busy() && recording) return;
  board.reset(); lastRecent = 'reset'; mic?.clearPending();
  if (worker) { epoch++; worker.postMessage({ type: 'reset', epoch }); setPhase('warming', 'Warming up…'); }
  renderScores(); renderStats();
}
function viewport() {
  document.body.style.setProperty('--viewport-height', `${window.visualViewport?.height ?? innerHeight}px`);
  renderPage();
}
async function boot() {
  $('start').onclick = () => start().catch(fail); $('stop').onclick = () => stop(); $('reset').onclick = reset;
  $('prev').onclick = () => { page--; renderPage(); }; $('next').onclick = () => { page++; renderPage(); };
  $('filter').oninput = () => { page = 0; renderPage(); };
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    for (const details of document.querySelectorAll('details[open]')) { details.open = false; details.querySelector('summary').focus(); }
  });
  document.addEventListener('pointerdown', event => {
    for (const details of document.querySelectorAll('details[open]')) if (!details.contains(event.target)) details.open = false;
  });
  window.addEventListener('pagehide', () => { mic?.stop(); worker?.terminate(); });
  window.addEventListener('beforeunload', event => { if (recording && busy()) { event.preventDefault(); event.returnValue = ''; } });
  window.visualViewport?.addEventListener('resize', viewport);
  new ResizeObserver(renderPage).observe($('rows'));
  try {
    const response = await fetch('/api/releases');
    if (!response.ok) throw new Error(`Cannot load releases (${response.status}). Reload to retry.`);
    ({ releases } = await response.json());
    renderReleases(); selectReleases(); viewport();
  } catch (error) { fail(error); }
}
boot();
