import { Microphone, listInputs, trimToSpeech, wavBlob } from '/capture.mjs';
import { PACES, buildQueue, estimateSeconds, blockKey, skipPhrase, requeue, formatDuration } from '/queue.mjs';
// Owner collection page: pick a word (or all words), record what is still missing, listen back.
// The queue is derived from the take store, so a sitting can stop at any moment and the next one
// continues from what is saved. Space pauses (the microphone closes), S skips a phrase, R redoes
// the last take, Esc stops.

const $ = id => document.getElementById(id);
const sessionId = crypto.randomUUID();
// One Microphone per selected input device: the same phrase can be captured on several mics at
// once, so one sitting yields several acoustics of the same words.
let microphones = [new Microphone()];
let selectedMics = [{ deviceId: null, label: 'Default microphone' }];
const recordingNow = () => microphones.some(mic => mic.active);
// Device ids are long opaque hashes: never show them raw, and prefer the human name when known.
let knownInputs = [];
const deviceName = mic => mic.label || (knownInputs.find(d => (d.deviceId ?? null) === (mic.deviceId ?? null))?.label) || (mic.deviceId ? `${mic.deviceId.slice(0, 6)}…` : 'default');
const SETTINGS = 'waqy-collect-settings';
let protocol, word = null, language = 'de', takesPerPhrase = 3;
let coverage = {};               // word -> { positive, negative, languages }
let coverageTotal = 0;
let capacity = null;   // { count, bytes, maxCount, maxBytes } from the store summary
let have = {};                   // word|language|label|style|phrase -> saved takes (server summary)
let takesFilter = 'all';
let playing = null;   // { id, audio } — one take at a time
let reviewTakes = [];
let reviewScope = 'word';   // 'word' | 'all'
let pendingDelete = null;   // in-page confirmation; browser confirm() can be blocked
let category = 'both';      // positive (wake) | negative (look-alikes) | both
let recordScope = 'word';   // 'word' | 'all'
let pace = 'fast';
// Peak level is a review hint, not SNR or evidence of unusable speech. Save quiet, nonzero
// recordings too; only missing/digitally silent input stops capture. Never auto-normalize here.
const quietLimit = () => ($('style').value === 'normal' ? -30 : -40);
let run = null, revision = 0;
let savedCount = 0, takes = { download: null };

const say = text => { $('status').textContent = text; };
const fail = error => { $('error').textContent = 'Error: ' + (error?.message ?? String(error)); $('error').hidden = false; };
const clearError = () => { $('error').hidden = true; $('error').textContent = ''; };
const dbfs = peak => peak > 0 ? 20 * Math.log10(peak) : -Infinity;
const peakOf = pcm => { let peak = 0; for (const value of pcm) peak = Math.max(peak, Math.abs(value)); return peak; };
// Segmented groups are button rows with aria-pressed; one value is active at a time.
function setSegment(id, value) {
  for (const button of $(id).querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.value === value));
}
function onSegment(id, handler) {
  for (const button of $(id).querySelectorAll('button')) button.onclick = () => { if (!button.disabled) handler(button.dataset.value); };
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS, JSON.stringify({ language, category, recordScope, pace, takesPerPhrase, style: $('style').value, word: word?.name, mics: selectedMics })); } catch {}
}
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS) ?? '{}') ?? {}; } catch { return {}; }
}

function plan() {
  if (!protocol) return { items: [], extra: false };
  return buildQueue(protocol, { scope: recordScope, word: word?.name, language, category, style: $('style').value, target: takesPerPhrase, have });
}
function renderPlan() {
  const { items } = plan();
  $('plan').textContent = items.slice(0, 400).map((item, i) => `${i + 1}. ${item.display} · ${item.phrase.line} · ${item.groupLabel} · take ${item.take}`).join('\n')
    + (items.length > 400 ? `\n… ${items.length - 400} more` : '');
  $('plan-count').textContent = `${items.length} takes · about ${formatDuration(estimateSeconds(items, pace))} at ${pace} pace`;
  if (!run) stage('idle', items, 0);
  $('word-title').textContent = recordScope === 'all' ? 'All words' : (word?.display ?? word?.name ?? '…');
  controls();
}
function stage(state, items, i) {
  const item = items[i];
  $('stage').dataset.state = state;
  const context = item ? `${item.display} · ${item.language.toUpperCase()} · ${item.category === 'positive' ? 'wake' : 'look-alike'}` : '';
  const label = { idle: 'First phrase', ready: 'Get ready', speak: 'Speak now', paused: 'Paused', done: 'Done' }[state];
  $('cue').textContent = item && state !== 'done' ? `${label} · ${context}` : label;
  $('phrase').textContent = item ? item.phrase.line : (state === 'done' ? 'Done' : 'All recorded');
  const style = { whispered: 'whisper it', far: 'say it from across the room' }[$('style').value];
  $('how').textContent = item ? [item.phrase.note, style].filter(Boolean).join(' · ') : '';
  const left = formatDuration(estimateSeconds(items.slice(Math.max(0, i)), pace));
  $('progress').textContent = !item ? `${items.length} takes` : state === 'idle' ? `${items.length.toLocaleString()} takes` : `Take ${i + 1} of ${items.length.toLocaleString()} · ${left} left`;
  const next = items[i + 1];
  $('next').textContent = state === 'idle' ? (items.length ? `about ${left}` : '')
    : state === 'paused' ? 'Space or Resume to continue'
    : next ? `Next: ${next.phrase.line}${next.word !== item?.word ? ` (${next.display})` : ''}` : (item ? 'Last one' : '');
  if (run && recordScope === 'all' && item) $('word-title').textContent = item.display;
  const fill = $('fill');
  fill.style.transition = 'none'; fill.style.width = '0%';
  if (state === 'speak') { void fill.offsetWidth; fill.style.transition = `width ${PACES[pace].window}ms linear`; fill.style.width = '100%'; }
}
function controls() {
  const busy = Boolean(run);
  const { items, extra } = busy ? { items: [], extra: false } : plan();
  $('start').disabled = !busy && !items.length;
  $('start').textContent = busy ? (run.paused ? 'Resume' : 'Pause')
    : items.length ? `Record ${items.length.toLocaleString()} ${extra ? 'more' : 'missing'} takes · ${formatDuration(estimateSeconds(items, pace))}` : 'Everything recorded';
  $('start').setAttribute('aria-keyshortcuts', busy ? 'Space' : '');
  $('skip').hidden = $('redo').hidden = $('keys').hidden = !busy;
  $('redo').disabled = !run?.saved.length;
  $('stop').disabled = !busy;
  for (const id of ['word', 'take-count', 'style', 'pace']) $(id).disabled = busy;
  for (const input of document.querySelectorAll('#mic-list input')) input.disabled = busy;
  for (const button of document.querySelectorAll('#language button, #label button, #record-scope button, #rail button')) {
    const unavailable = recordScope === 'word' && button.closest('#language') && !word?.languages?.[button.dataset.value];
    button.disabled = busy || Boolean(unavailable);
  }
  $('delete').disabled = !$('saved-id').dataset.id;
  $('download').disabled = takes.download === null;
}
async function save(pcm, item, mic = null) {
  const body = new FormData();
  // Several mics save in parallel: one take per device, same phrase, trimmed to its own speech span.
  body.set('audio', wavBlob(pcm), `${item.word}-${mic ? mic.deviceId ?? 'default' : 'default'}.wav`);
  body.set('meta', JSON.stringify({ sessionId, language: item.language, group: item.groupId, label: item.category, word: item.word,
    form: item.phrase.form, take: item.take, phrase: item.phrase.line, spoken: item.phrase.spoken, words: item.phrase.note,
    style: item.style, mic: mic ? deviceName(mic) : 'default microphone', micDevice: mic ? mic.deviceId : null }));
  const response = await fetch('/api/recordings', { method: 'POST', credentials: 'same-origin', cache: 'no-store', body,
    signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error(`${response.status} ${(await response.text()).slice(0, 200)}`);
  return response.json();
}
function showLevel(level, quiet, perDevice = '') {
  $('level').hidden = false;
  $('level').dataset.quiet = String(quiet);
  const value = Number.isFinite(level) ? level.toFixed(0) : '−∞';
  $('level').textContent = (quiet ? `Last take ${value} dBFS: low level; listen back when reviewing.` : `Last take ${value} dBFS: level OK.`)
    + (perDevice ? ` (${perDevice})` : '');
}

// Interruptible wait: resolves null after `ms`, or early with the pending interrupt reason.
function wait(ms) {
  return new Promise(resolve => {
    if (run?.reason) { resolve(run.reason); return; }
    const timer = setTimeout(() => { if (run) run.wake = null; resolve(null); }, ms);
    run.wake = reason => { clearTimeout(timer); run.wake = null; resolve(reason); };
  });
}
function interrupt(reason) {
  if (!run) return;
  run.reason = reason;
  run.wake?.(reason);
  run.resume?.();
}
function togglePause() {
  if (!run) return;
  if (run.paused) { run.paused = false; run.pauseMessage = null; run.resume?.(); say('Resumed.'); }
  else interrupt('pause');
  controls();
}
async function redoLast() {
  // Deletes the last saved take and puts it back at the front of the queue.
  if (!run) return;
  await Promise.allSettled([...run.pending]);
  const last = run.saved.pop();
  if (!last) { say('Nothing to redo yet.'); controls(); return; }
  const response = await fetch(`/api/recordings/${last.id}`, { method: 'DELETE', credentials: 'same-origin' });
  if (!response.ok) { run.saved.push(last); say(`Could not delete ${last.id}: ${response.status}`); return; }
  have[last.item.key] = Math.max(0, (have[last.item.key] ?? 1) - 1);
  savedCount = Math.max(0, savedCount - 1);
  run.items = requeue(run.items, run.index, last.item);
  $('log').textContent += `redo: deleted ${last.id}\n`;
  say(`Redo: “${last.item.phrase.line}” deleted, record it again.`);
  controls();
}
async function handle(reason) {
  run.reason = null;
  if (reason === 'pause') run.paused = true;
  else if (reason === 'skip') { const line = run.items[run.index]?.phrase.line; run.items = skipPhrase(run.items, run.index); say(`Skipped “${line}” for this sitting.`); }
  else if (reason === 'redo') await redoLast();
  else if (reason === 'stop') run.reason = 'stop';
  controls();
}

async function record() {
  const { items } = plan();
  if (!items.length) return;
  const token = ++revision;
  run = { token, items, index: 0, paused: false, pauseMessage: null, reason: null, wake: null, resume: null, quiet: 0, saved: [], pending: new Set(), block: null };
  clearError(); $('level').hidden = true; document.activeElement?.blur?.(); saveSettings(); controls();
  say('Recording. Say each phrase when the card turns to “Speak now”.');
  let first = true;
  try {
    while (run?.token === token && run.index < run.items.length && run.reason !== 'stop') {
      if (run.paused) {
        for (const mic of microphones) mic.cancel();
        run.block = null;
        say(run.pauseMessage ?? 'Paused. The microphone is closed; Space or Resume continues.');
        stage('paused', run.items, run.index); controls();
        await new Promise(resolve => { run.resume = resolve; });
        run.resume = null; first = true;
        continue;
      }
      const item = run.items[run.index];
      const p = PACES[pace];
      if (!recordingNow() || run.block !== blockKey(item)) {
        // One capture block per word/language/category keeps each device's buffer small.
        for (const mic of microphones) mic.cancel();
        let length = 0;
        while (run.index + length < run.items.length && blockKey(run.items[run.index + length]) === blockKey(item)) length++;
        const seconds = Math.ceil(length * (p.cue + p.window + p.gap + 800) / 1000) + 60;
        const started = await Promise.allSettled(microphones.map(mic => mic.start({ maxSeconds: seconds, deviceId: mic.deviceId })));
        microphones = microphones.filter((mic, i) => started[i].status === 'fulfilled' && started[i].value);
        if (!microphones.length) { fail(new Error('No microphone could be opened.')); break; }
        const failed = started.filter(r => r.status === 'fulfilled' && !r.value).length + started.filter(r => r.status === 'rejected').length;
        if (failed) fail(new Error(`${failed} microphone(s) could not be opened; recording on ${microphones.length}.`));
        if (run?.token !== token) break;
        run.block = blockKey(item); first = true;
      }
      stage('ready', run.items, run.index);
      let reason = await wait(first ? p.first : p.cue);
      first = false;
      if (reason) { await handle(reason); continue; }
      stage('speak', run.items, run.index);
      const from = microphones.map(mic => mic.interim().length / 16000);
      reason = await wait(p.window);
      if (reason) { await handle(reason); continue; }   // an interrupted take is discarded
      // Per device: its own window, cut to the phrase so devices that start later (Bluetooth) align.
      // perMic: one trimmed slice per device for this take (not the download list below).
      const perMic = microphones.map((mic, i) => {
        const slice = mic.interimSlice(from[i]);
        return { mic, pcm: trimToSpeech(slice), peak: peakOf(slice) };
      });
      if (!perMic.some(take => take.pcm.length && take.peak > 0)) {
        run.paused = true;
        run.pauseMessage = 'Paused: no audio from any microphone (missing or digitally silent input). This take was not saved; check the input, then Resume.';
        continue;
      }
      const silent = perMic.filter(take => !take.pcm.length || take.peak === 0);
      if (silent.length) say(`${silent.length} microphone(s) heard nothing this take: ${silent.map(t => deviceName(t.mic)).join(', ')}.`);
      const live = perMic.filter(take => take.pcm.length && take.peak > 0);
      const level = dbfs(Math.max(...live.map(take => take.peak)));
      const quiet = level < quietLimit();
      if (quiet) run.quiet++;
      showLevel(level, quiet, live.map(take => `${deviceName(take.mic)}: ${dbfs(take.peak).toFixed(0)}`).join(' · '));
      run.index++;
      const current = run;
      const saving = Promise.all(live.map(take => save(take.pcm, item, take.mic).then(result => {
        savedCount++;
        current.saved.push({ id: result.id, item });
        have[item.key] = (have[item.key] ?? 0) + 1;
        const bucket = (coverage[item.word] ??= { positive: 0, negative: 0, languages: {} });
        bucket[item.category === 'negative' ? 'negative' : 'positive'] += 1;
        bucket.languages[item.language] = (bucket.languages[item.language] ?? 0) + 1;
        coverageTotal += 1;
        $('saved-id').dataset.id = result.id;
        $('log').textContent += `${item.display} · ${item.phrase.line} · take ${item.take} · ${deviceName(take.mic)} · ${dbfs(take.peak).toFixed(0)} dBFS · saved ${result.id}\n`;
        takes.download = takes.download ?? [];
        takes.download.push({ ...item, label: item.category, spoken: item.phrase.spoken, id: result.id, mic: deviceName(take.mic), seconds: take.pcm.length / 16000, peakDbfs: dbfs(take.peak) });
        renderRail();
      }))).catch(error => {
        // A failed upload must never generate an endless re-record loop. Keep the item missing
        // and close the microphones at the next interrupt boundary; retry only on explicit Resume.
        const message = error?.message ?? String(error);
        if (current === run) {
          current.items = requeue(current.items, current.index, item);
          current.pauseMessage = `Paused: could not save “${item.phrase.line}” (${message}). Check the connection or storage, then Resume to retry.`;
          if (!current.paused && current.reason !== 'stop') interrupt('pause');
        }
        $('log').textContent += `${item.display} · ${item.phrase.line} · NOT saved: ${message}\n`;
        fail(new Error(message));
      }).finally(() => current.pending.delete(saving));
      current.pending.add(saving);
      reason = await wait(p.gap);
      if (reason) await handle(reason);
    }
  } catch (error) { if (run?.token === token) fail(error); }
  finally {
    for (const mic of microphones) mic.cancel();
    if (run?.token === token) {
      const done = run;
      await Promise.allSettled([...done.pending]);
      run = null;
      const remaining = plan();
      stage('done', remaining.items, -1);
      $('next').textContent = '';
      say(`${done.saved.length} takes saved this sitting${done.quiet ? `, ${done.quiet} low-level takes flagged for review` : ''}. `
        + (remaining.items.length && !remaining.extra ? `${remaining.items.length.toLocaleString()} still missing; press Record to continue, now or later.` : 'Nothing missing in this selection.'));
      renderPlan(); void loadCoverage(); void loadTakes();
    }
  }
}
function stop() {
  for (const mic of microphones) mic.cancel();
  if (!run) return;
  interrupt('stop');
}
async function loadCoverage() {
  try {
    // The take listing is capped for the UI; counts must use whole-store aggregates.
    const response = await fetch('/api/recordings/summary', { cache: 'no-store' });
    if (!response.ok) throw new Error(String(response.status));
    const summary = await response.json();
    coverage = summary.byWord ?? {};
    have = summary.byPhrase ?? {};
    capacity = summary.capacity ?? null;
    coverageTotal = summary.total ?? 0;
    renderRail(); if (!run) renderPlan();
  } catch (error) {
    $('rail-total').textContent = `counts unavailable: ${error.message}`;
  }
}
// Word rail: recorded words first (most takes on top), then the untouched ones alphabetically.
function renderRail() {
  const count = entry => (coverage[entry.name]?.positive ?? 0) + (coverage[entry.name]?.negative ?? 0);
  const words = [...protocol.words].sort((a, b) => (count(b) - count(a)) || (a.display ?? a.name).localeCompare(b.display ?? b.name));
  const started = words.filter(entry => count(entry) > 0);
  const nodes = [];
  const group = text => Object.assign(document.createElement('li'), { className: 'rail-group', textContent: text });
  if (started.length) nodes.push(group('Recorded'));
  words.forEach((entry, index) => {
    if (index === started.length) nodes.push(group(started.length ? 'Not started' : 'All words'));
    const item = document.createElement('li');
    const button = document.createElement('button'); button.type = 'button'; button.className = 'rail-word';
    button.dataset.word = entry.name;
    if (entry.name === word?.name) button.setAttribute('aria-current', 'true');
    const wake = coverage[entry.name]?.positive ?? 0, look = coverage[entry.name]?.negative ?? 0;
    const name = Object.assign(document.createElement('span'), { className: 'rail-name', textContent: entry.display ?? entry.name });
    const counts = Object.assign(document.createElement('span'), { className: 'rail-counts', textContent: count(entry) ? `${wake} · ${look}` : '' });
    counts.title = `${wake} wake, ${look} look-alike takes`;
    button.setAttribute('aria-label', `${entry.display ?? entry.name}: ${wake} wake, ${look} look-alike takes`);
    button.append(name, counts);
    button.onclick = () => selectWord(entry.name);
    item.append(button); nodes.push(item);
  });
  $('rail').replaceChildren(...nodes);
  $('rail-total').textContent = `${coverageTotal} takes · wake · look`;
  controls();
}
function selectWord(name, { keepScope = false } = {}) {
  if (run) return;
  if (!keepScope && recordScope === 'all') { recordScope = 'word'; setSegment('record-scope', 'word'); }
  pendingDelete = null; stopPlayback();
  word = protocol.words.find(entry => entry.name === name) ?? word;
  $('word').value = word.name;
  $('word-title').textContent = word.display ?? word.name;
  $('scope-word').textContent = word.display ?? word.name;
  $('record-word').textContent = word.display ?? word.name;
  // Jump straight to this word's look-alike list from the recorder.
  $('nav-lookalikes').href = `/lookalikes#w-${word.name}`;
  $('level').hidden = true;
  setLanguage(language); renderRail(); saveSettings(); void loadTakes();
}

function stopPlayback() {
  if (!playing) return;
  playing.audio.pause();
  URL.revokeObjectURL(playing.audio.src);
  playing = null;
  renderTakes();
}
function renderTakes() {
  const root = $('takes');
  const visible = reviewTakes.filter(t => takesFilter === 'all' || t.label === takesFilter);
  $('review-count').textContent = visible.length === reviewTakes.length ? `${reviewTakes.length}` : `${visible.length} of ${reviewTakes.length}`;
  if (capacity) {
    const mib = n => `${(n / 1024 ** 2).toFixed(0)} MB`;
    const share = Math.max(capacity.count / capacity.maxCount, capacity.bytes / capacity.maxBytes);
    $('review-count').title = `${capacity.count.toLocaleString()} of ${capacity.maxCount.toLocaleString()} takes; ${mib(capacity.bytes)} of ${mib(capacity.maxBytes)}`;
    if (share >= 0.9) $('review-count').textContent += ` · store ${Math.round(share * 100)}% full`;
  }
  if (!visible.length) {
    root.replaceChildren(Object.assign(document.createElement('li'), { className: 'take-empty',
      textContent: reviewTakes.length ? 'No takes in this filter.' : 'No owner takes saved for this word yet.' }));
    return;
  }
  root.replaceChildren(...visible.map(take => {
    const row = document.createElement('li'); row.className = 'take'; row.dataset.id = take.id;
    row.dataset.playing = String(playing?.id === take.id);
    row.dataset.pending = String(pendingDelete === take.id);
    const when = document.createElement('span'); when.className = 'take-when';
    const at = new Date(take.receivedAt);
    const today = at.toDateString() === new Date().toDateString();
    when.textContent = today ? at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : at.toLocaleDateString([], { month: 'short', day: 'numeric' });
    when.title = at.toLocaleString();
    const what = document.createElement('span'); what.className = 'take-what';
    // peakPcm is stored on the int16 scale; normalize before converting to dBFS.
    const peak = Number.isFinite(take.peakPcm) ? take.peakPcm / (take.peakPcm > 1 ? 32768 : 1) : null;
    const level = peak && peak > 0 ? 20 * Math.log10(peak) : null;
    const quiet = level !== null && level < -30;
    const label = document.createElement('strong'); label.textContent = take.phrase;
    if (reviewScope === 'all') label.append(Object.assign(document.createElement('em'), { textContent: ` ${take.word ?? ''}` }));
    const meta = document.createElement('small');
    const mic = take.mic && take.mic !== 'default microphone' ? ` · ${take.mic}` : '';
    meta.textContent = `${take.language.toUpperCase()} · ${take.label === 'negative' ? 'look-alike' : 'wake'} · ${take.form ?? take.label} · ${take.style ?? 'normal'}${mic} · ${take.seconds.toFixed(2)} s · ${level === null ? 'level ?' : `${level.toFixed(0)} dBFS`}${quiet ? ' · QUIET' : ''}`;
    if (quiet) meta.style.color = 'var(--warning)';
    what.append(label, meta);
    const play = document.createElement('button'); play.type = 'button';
    play.textContent = playing?.id === take.id ? 'Stop' : 'Play';
    play.setAttribute('aria-label', `${play.textContent} ${take.phrase}`);
    play.onclick = async () => {
      if (playing?.id === take.id) { stopPlayback(); return; }
      stopPlayback();
      const audio = new Audio(`/api/recordings/${take.id}/audio`);
      playing = { id: take.id, audio };
      audio.onended = stopPlayback; audio.onerror = () => { stopPlayback(); say('Could not play that take.'); };
      renderTakes();
      try { await audio.play(); } catch { stopPlayback(); say('Could not play that take.'); }
    };
    // In-page two-step delete: browser dialogs can be blocked per-site, which would silently
    // cancel every deletion, and an accessible confirmation beats a native popup anyway.
    const actions = document.createElement('span'); actions.className = 'take-actions';
    if (pendingDelete === take.id) {
      const confirmButton = document.createElement('button'); confirmButton.type = 'button'; confirmButton.className = 'delete confirm';
      confirmButton.textContent = 'Delete for good';
      confirmButton.onclick = async () => {
        if (playing?.id === take.id) stopPlayback();
        const response = await fetch(`/api/recordings/${take.id}`, { method: 'DELETE', credentials: 'same-origin' });
        pendingDelete = null;
        if (!response.ok) { say(`Could not delete ${take.id}: ${response.status}`); renderTakes(); return; }
        reviewTakes = reviewTakes.filter(entry => entry.id !== take.id);
        renderTakes(); void loadCoverage();
        say(`Deleted ${take.phrase} (${take.id}).`);
      };
      const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel';
      cancel.onclick = () => { pendingDelete = null; renderTakes(); };
      actions.append(confirmButton, cancel);
    } else {
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'delete';
      remove.textContent = 'Delete';
      remove.onclick = () => { pendingDelete = take.id; renderTakes(); };
      actions.append(remove);
    }
    row.append(when, what, play, actions);

    return row;
  }));
}
async function loadTakes() {
  if (!word) return;
  try {
    const scope = reviewScope === 'all' ? 'all' : encodeURIComponent(word.name);
    const response = await fetch(`/api/recordings/takes/${scope}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(String(response.status));
    reviewTakes = (await response.json()).takes ?? [];
    renderTakes();
  } catch (error) {
    $('takes').replaceChildren(Object.assign(document.createElement('li'), { className: 'take-empty', textContent: `Could not list takes: ${error.message}` }));
  }
}

async function boot() {
  if (!isSecureContext || !crossOriginIsolated) {
    fail(new Error('Open the private HTTPS link. This page needs a secure, cross-origin-isolated context for local capture.'));
    $('start').disabled = true; return;
  }
  try {
    protocol = await (await fetch('/wordpack.json', { cache: 'no-store' })).json();
    $('word').replaceChildren(...protocol.words.map(entry => Object.assign(document.createElement('option'), { value: entry.name, textContent: entry.display ?? entry.name })));
    // Resume where the last sitting left off: same selection, same pace; the queue itself comes from the store.
    const saved = loadSettings();
    if (['de', 'en'].includes(saved.language)) language = saved.language;
    if (['positive', 'negative', 'both'].includes(saved.category)) category = saved.category;
    if (['word', 'all'].includes(saved.recordScope)) recordScope = saved.recordScope;
    if (PACES[saved.pace]) pace = saved.pace;
    if ([1, 2, 3, 6].includes(saved.takesPerPhrase)) takesPerPhrase = saved.takesPerPhrase;
    if (['normal', 'whispered', 'far'].includes(saved.style)) $('style').value = saved.style;
    $('pace').value = pace; $('take-count').value = String(takesPerPhrase);
    setSegment('label', category); setSegment('record-scope', recordScope);
    if (Array.isArray(saved.mics) && saved.mics.length) selectedMics = saved.mics.filter(mic => mic && typeof mic.label === 'string');
    microphones = selectedMics.map(mic => Object.assign(new Microphone(), { deviceId: mic.deviceId ?? null, label: mic.label }));
    void renderMics();
    word = protocol.words.find(entry => entry.name === saved.word) ?? protocol.words.find(entry => entry.name === 'hex') ?? protocol.words[0];
    selectWord(word.name, { keepScope: true }); void loadCoverage();
  } catch (error) { fail(error); }
}
function setLanguage(id) {
  language = id;
  if (recordScope === 'word' && !word?.languages?.[id]) {
    const other = ['de', 'en'].find(lang => word?.languages?.[lang]);
    if (other) language = other;
  }
  setSegment('language', language);
  renderPlan();
}
$('word').onchange = () => selectWord($('word').value);
for (const button of document.querySelectorAll('.filter')) button.onclick = () => {
  takesFilter = button.dataset.filter;
  for (const sibling of document.querySelectorAll('.filter')) sibling.setAttribute('aria-pressed', String(sibling === button));
  renderTakes();
};
onSegment('review-scope', value => { reviewScope = value; setSegment('review-scope', value); pendingDelete = null; stopPlayback(); void loadTakes(); });
onSegment('record-scope', value => { recordScope = value; setSegment('record-scope', value); setLanguage(language); saveSettings(); });
onSegment('language', value => { setLanguage(value); saveSettings(); });
onSegment('label', value => { category = value; setSegment('label', value); renderPlan(); saveSettings(); });
$('take-count').onchange = () => { takesPerPhrase = Number($('take-count').value); renderPlan(); saveSettings(); };
$('pace').onchange = () => { pace = $('pace').value; renderPlan(); saveSettings(); };
$('style').onchange = () => { renderPlan(); saveSettings(); };
$('start').onclick = async () => {
  if (run) { togglePause(); return; }
  try { await record(); } catch (error) { fail(error); stop(); }
};
$('skip').onclick = () => { if (!run) return; if (run.paused) void handle('skip').then(() => stage('paused', run.items, run.index)); else interrupt('skip'); };
$('redo').onclick = () => { if (!run) return; if (run.paused) void redoLast().then(() => run && stage('paused', run.items, run.index)); else interrupt('redo'); };
$('stop').onclick = stop;
document.addEventListener('keydown', event => {
  if (!run || event.metaKey || event.ctrlKey || event.altKey || event.target.closest?.('select, input, textarea')) return;
  const key = event.key.toLowerCase();
  if (key === ' ') { event.preventDefault(); togglePause(); }
  else if (key === 's') { event.preventDefault(); $('skip').click(); }
  else if (key === 'r' || key === 'backspace') { event.preventDefault(); $('redo').click(); }
  else if (key === 'escape') { event.preventDefault(); stop(); }
});
$('download').onclick = () => {
  if (!takes.download) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify({ sessionId, takes: takes.download }, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = 'wake-takes.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('delete').onclick = async () => {
  const id = $('saved-id').dataset.id;
  if (!id) return;
  const response = await fetch('/api/recordings/' + id, { method: 'DELETE', credentials: 'same-origin' });
  if (!response.ok) { say(`Could not delete ${id}: ${response.status}`); return; }
  $('saved-id').dataset.id = ''; $('saved-id').textContent = ''; $('delete').disabled = true;
  savedCount = Math.max(0, savedCount - 1); $('log').textContent += `deleted ${id}\n`;
  say(`Deleted the last take (${id}) from this host.`); void loadCoverage();
};
// Mics: pick which inputs record at once. Labels only appear after microphone permission.
async function renderMics() {
  let inputs = [];
  try { inputs = await listInputs(); } catch { inputs = []; }
  if (!inputs.length) inputs = [{ deviceId: null, label: 'Default microphone' }];
  knownInputs = inputs;
  // A stored label that is really a device hash (saved before permission existed) is replaced.
  const isHash = label => typeof label === 'string' && /^[A-Za-z0-9+/=_-]{20,}$/.test(label);
  selectedMics = selectedMics.map(mic => {
    const match = inputs.find(device => (device.deviceId ?? null) === (mic.deviceId ?? null) && !/^Microphone \d+$/.test(device.label));
    return mic.label && !isHash(mic.label) ? mic : { ...mic, label: match?.label ?? (mic.deviceId ? `${mic.deviceId.slice(0, 6)}…` : 'Default microphone') };
  });
  microphones = microphones.map((mic, i) => Object.assign(mic, { label: selectedMics[i]?.label }));
  const chosen = new Map(selectedMics.map(mic => [mic.deviceId ?? '', mic]));
  const rows = inputs.map(device => {
    const key = device.deviceId ?? '';
    const current = chosen.get(key);
    // A saved pick whose device is gone stays listed and stays unticked, so it is visible.
    const item = Object.assign(document.createElement('li'), { className: 'mic-row' });
    const label = Object.assign(document.createElement('label'), { className: 'check' });
    const box = Object.assign(document.createElement('input'), { type: 'checkbox', checked: Boolean(current) });
    box.dataset.device = key;
    box.onchange = () => {
      let next = selectedMics.filter(mic => (mic.deviceId ?? '') !== key);
      // Choosing a real device replaces the implicit default placeholder: otherwise the same
      // microphone records twice, once as "default".
      if (box.checked && key !== '') next = next.filter(mic => (mic.deviceId ?? '') !== '' || mic.label !== 'Default microphone');
      selectedMics = box.checked ? [...next, { deviceId: device.deviceId ?? null, label: device.label }] : next;
      if (!selectedMics.length) { box.checked = true; selectedMics = [{ deviceId: device.deviceId ?? null, label: device.label }]; fail(new Error('At least one microphone must stay selected.')); }
      microphones = selectedMics.map(mic => Object.assign(new Microphone(), { deviceId: mic.deviceId ?? null, label: mic.label }));
      saveSettings();
      $('mics-state').textContent = `${selectedMics.length} selected: ${selectedMics.map(mic => mic.label).join(', ')}`;
    };
    label.append(box, Object.assign(document.createElement('span'), { textContent: device.label }));
    item.append(label);
    return item;
  });
  for (const missing of chosen.values()) {
    if (!inputs.some(device => (device.deviceId ?? '') === (missing.deviceId ?? ''))) {
      const item = Object.assign(document.createElement('li'), { className: 'mic-row mic-missing' });
      item.textContent = `${missing.label}: not connected (unticked)`;
      rows.push(item);
    }
  }
  // Device names stay hidden until the page holds microphone permission once; offer that step here
  // instead of leaving the owner guessing why every device is called "Microphone 1".
  const hidden = inputs.every(device => /^Microphone \d+$/.test(device.label)) || !inputs.some(device => device.label);
  if (hidden) {
    const allow = Object.assign(document.createElement('button'), { type: 'button', textContent: 'Allow microphones', className: 'mic-allow' });
    allow.onclick = async () => {
      allow.disabled = true;
      try {
        // One permission prompt per site covers every input; this just triggers it and releases it.
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
        await renderMics();
      } catch (error) {
        fail(new Error(`Microphone permission: ${error.message}. In the browser's site settings set Microphone to Allow, then reload.`));
      } finally { allow.disabled = false; }
    };
    const note = Object.assign(document.createElement('li'), { className: 'mic-row mic-note' });
    note.append(Object.assign(document.createElement('span'), { textContent: 'Names appear after microphone permission: ' }), allow);
    rows.unshift(note);
  }
  $('mic-list').replaceChildren(...rows);
  $('mics-state').textContent = selectedMics.length === 1 ? `${selectedMics[0].label}` : `${selectedMics.length} mics at once: ${selectedMics.map(mic => mic.label).join(', ')}`;
}

void boot();
window.addEventListener('pagehide', () => { ++revision; for (const mic of microphones) mic.cancel(); stopPlayback(); });
window.wakeCollect = { refreshMics: () => renderMics(), mics: () => microphones.map(mic => ({ device: mic.deviceId ?? 'default', active: mic.active })), plan: () => plan().items, takes: () => reviewTakes, playing: () => playing?.id ?? null,
  saved: () => ({ count: savedCount, download: takes.download }), active: () => recordingNow(), coverage: () => coverage,
  run: () => run && { index: run.index, length: run.items.length, paused: run.paused, saved: run.saved.map(entry => entry.id), quiet: run.quiet,
                      current: run.items[run.index]?.key ?? null } };
