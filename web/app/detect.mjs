import { PcmResampler } from '/resampler.js';
import { wavBlob } from '/capture.mjs';

const $ = (id) => document.getElementById(id);
const LABELS = {
  hey_oqto: 'Hey Oqto', oqto: 'Oqto',
  // comparison release: same audio scored by the shipped and the new heads
  hey_r2: 'Hey Oqto · shipped', oqto_r2: 'Oqto · shipped',
  hey_new: 'Hey Oqto · new', oqto_new: 'Oqto · new',
};
const CHUNK = 16000 * 10; // samples per uploaded chunk; must match live.mjs CHUNK_SECONDS
let manifest = null;
let worker = null;
let mic = null;
let recorder = null;
let frames = 0;
let msTotal = 0;
let lastDetection = null;
let day = null;
const log = [];

function status(text) { $('status').textContent = text; }
function fail(error) {
  $('error').hidden = false;
  $('error').textContent = String(error?.message ?? error);
}

/** Continuous microphone at 16 kHz. */
class LiveMicrophone {
  async start(onPcm) {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Open the HTTPS link to use the microphone.');
    this.context = new AudioContext();
    await this.context.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
    await this.context.audioWorklet.addModule('/capture-worklet.js');
    const resampler = new PcmResampler(this.context.sampleRate, 16000);
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, 'moonshine-capture');
    this.mute = this.context.createGain();
    this.mute.gain.value = 0;
    this.node.port.onmessage = ({ data }) => { const pcm = resampler.feed(data); if (pcm.length) onPcm(pcm.slice()); };
    this.source.connect(this.node).connect(this.mute).connect(this.context.destination);
  }
  stop() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.node?.disconnect();
    this.context?.close();
  }
}

/** Saves the session privately on this host: 10 s WAV chunks plus the scores computed over them. */
class LiveRecorder {
  constructor() {
    this.sessionId = crypto.randomUUID();
    this.parts = [];
    this.buffered = 0;
    this.samples = 0; // total samples captured, = position for marks
    this.startSample = 0;
    this.seq = 0;
    this.scores = [];
    this.saved = 0;
    this.failed = 0;
    this.queue = Promise.resolve();
  }
  push(pcm) {
    this.parts.push(pcm);
    this.buffered += pcm.length;
    this.samples += pcm.length;
    if (this.buffered >= CHUNK) this.flush();
  }
  score(row) { this.scores.push(row); }
  flush() {
    if (!this.buffered) return this.queue;
    const pcm = new Float32Array(this.buffered);
    let offset = 0;
    for (const part of this.parts) { pcm.set(part, offset); offset += part.length; }
    const meta = { sessionId: this.sessionId, seq: this.seq++, startSample: this.startSample, release: manifest.release,
      threshold: Number($('threshold').value), scores: this.scores.splice(0) };
    this.startSample += pcm.length;
    this.parts = [];
    this.buffered = 0;
    const body = new FormData();
    body.set('audio', wavBlob(pcm), 'chunk.wav');
    body.set('meta', JSON.stringify(meta));
    this.queue = this.queue.then(async () => {
      for (let attempt = 0; attempt < 2; attempt++) {
        const response = await fetch('/api/live/chunk', { method: 'POST', credentials: 'same-origin', cache: 'no-store', body }).catch((e) => ({ ok: false, statusText: e.message }));
        if (response.ok) { this.saved += pcm.length; return renderRecording(); }
      }
      this.failed++;
      fail(`A recording chunk could not be saved (${this.failed} so far). Detection keeps running.`);
    });
    return this.queue;
  }
  mark(kind, atSample, detection = null) {
    return fetch('/api/live/mark', { method: 'POST', credentials: 'same-origin', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: this.sessionId, kind, atSample, detection }) });
  }
}

/** All-day mode: a rolling 10 s buffer; only ~4.5 s around each detection or "Missed it" press is saved. */
class DayLog {
  constructor() {
    this.sessionId = crypto.randomUUID();
    this.ring = new Float32Array(16000 * 10);
    this.samples = 0;
    this.pending = []; // { kind, atSample, endSample, model, score }
    this.seq = 0;
    this.lastSnippetAt = -Infinity;
    this.detections = []; // { atSample, model, score, verdict, when }
    this.pausedSeconds = 0;
    this.saved = 0;
    this.failed = 0;
    this.timer = setInterval(() => this.heartbeat(), 60000);
  }
  push(pcm) {
    for (let i = 0; i < pcm.length; i++) this.ring[(this.samples + i) % this.ring.length] = pcm[i];
    this.samples += pcm.length;
    const due = this.pending.filter((p) => this.samples >= p.endSample);
    this.pending = this.pending.filter((p) => this.samples < p.endSample);
    for (const p of due) this.save(p);
  }
  cut(start, end) {
    start = Math.max(start, this.samples - this.ring.length, 0);
    const out = new Float32Array(Math.max(0, end - start));
    for (let i = 0; i < out.length; i++) out[i] = this.ring[(start + i) % this.ring.length];
    return { pcm: out, start };
  }
  detection(atSample, model, score) {
    // "Hey Okto" contains "Okto", so both heads fire on one utterance: keep one row for it
    const last = this.detections[0];
    if (last && Math.abs(last.atSample - atSample) < 16000 * 1.2 && last.model !== model) {
      last.also = [...new Set([...(last.also ?? []), model])];
      last.score = Math.max(last.score, score);
      return last;
    }
    const row = { atSample, model, score, verdict: null, when: new Date(), also: [] };
    this.detections.unshift(row);
    if (atSample - this.lastSnippetAt > 16000 * 2) { // one snippet per burst of detections
      this.lastSnippetAt = atSample;
      this.pending.push({ kind: 'detection', atSample, endSample: atSample + 16000 * 1.5, model, score });
    }
    return row;
  }
  missed() {
    const atSample = this.samples;
    this.save({ kind: 'missed', atSample, endSample: atSample, model: null, score: null }, 16000 * 5);
    return atSample;
  }
  async save(p, before = 16000 * 3) {
    const { pcm, start } = this.cut(p.atSample - before, p.endSample);
    const body = new FormData();
    body.set('audio', wavBlob(pcm), 'snippet.wav');
    body.set('meta', JSON.stringify({ sessionId: this.sessionId, seq: this.seq++, kind: p.kind, atSample: p.atSample, startSample: start,
      model: p.model, score: p.score, threshold: Number($('threshold').value), listenedSeconds: this.samples / 16000, release: manifest.release }));
    const response = await fetch('/api/live/snippet', { method: 'POST', credentials: 'same-origin', cache: 'no-store', body }).catch(() => ({ ok: false }));
    if (response.ok) this.saved++; else this.failed++;
    renderRecording();
  }
  mark(kind, row) {
    return fetch('/api/live/mark', { method: 'POST', credentials: 'same-origin', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: this.sessionId, kind, atSample: row.atSample, detection: row.model }) });
  }
  heartbeat() {
    return fetch('/api/live/heartbeat', { method: 'POST', credentials: 'same-origin', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: this.sessionId, listenedSeconds: this.samples / 16000, pausedSeconds: this.pausedSeconds,
        detections: this.detections.length, threshold: Number($('threshold').value), release: manifest.release }) }).catch(() => null);
  }
  async close() {
    clearInterval(this.timer);
    for (const p of this.pending.splice(0)) await this.save(p);
    await this.heartbeat();
  }
}

function renderReview() {
  const box = $('review');
  if (!day) return;
  const hours = Math.max(day.samples / 16000 / 3600, 1e-9);
  const wrong = day.detections.filter((d) => d.verdict === 'false-trigger').length;
  const right = day.detections.filter((d) => d.verdict === 'correct').length;
  $('tally').textContent = `${(day.samples / 16000 / 60).toFixed(0)} min listened · ${day.detections.length} detections · ${right} right · ${wrong} wrong`
    + (day.samples > 16000 * 600 ? ` · ${(wrong / hours).toFixed(2)} false wakes per hour` : '');
  box.replaceChildren(...day.detections.slice(0, 40).map((row) => {
    const item = document.createElement('div');
    item.className = 'review-row';
    const label = document.createElement('span');
    const both = row.also?.length ? ` + ${row.also.map((m) => LABELS[m] ?? m).join(' + ')}` : '';
    label.textContent = `${row.when.toLocaleTimeString()} · ${LABELS[row.model] ?? row.model}${both} · ${row.score.toFixed(2)}`;
    item.append(label);
    if (row.verdict) {
      const v = document.createElement('span'); v.className = 'verdict'; v.textContent = row.verdict === 'correct' ? 'right' : 'wrong'; item.append(v);
    } else {
      for (const [kind, text] of [['correct', 'Right'], ['false-trigger', 'Wrong']]) {
        const b = document.createElement('button'); b.textContent = text;
        b.onclick = async () => { const r = await day.mark(kind, row); if (r.ok) { row.verdict = kind; renderReview(); } };
        item.append(b);
      }
    }
    return item;
  }));
}

function renderRecording() {
  const mode = $('mode').value;
  if (day) { $('recording').textContent = `All-day mode: ${day.saved} snippets saved on this host${day.failed ? `, ${day.failed} failed` : ''}. Keep this tab open; closing the laptop stops listening.`; return; }
  if (!recorder) { $('recording').textContent = { off: 'Nothing is saved.', allday: 'All-day mode starts with Start: only a few seconds around each detection are saved.', full: 'Full recording starts with Start.' }[mode]; return; }
  const minutes = (recorder.saved / 16000 / 60).toFixed(1);
  $('recording').textContent = `Recording: ${minutes} min saved on this host · session ${recorder.sessionId.slice(0, 8)}`;
}

// Heads to test live: the installed release by default, or any published pack with a subset of its
// heads ticked. Picking a subset keeps the shared front end once and loads only what you test.
let packs = [];
async function renderPicker() {
  const box = $('release');
  try {
    packs = (await (await fetch('/api/releases', { cache: 'no-store' })).json()).releases ?? [];
  } catch { packs = []; }
  const options = [{ label: `Installed (${manifest.release})`, value: '', heads: manifest.classifiers }]
    .concat(packs.map(pack => ({ label: `${pack.release} (${pack.classifiers.length} heads)`, value: pack.release, heads: pack.classifiers })));
  box.replaceChildren(...options.map(option => Object.assign(document.createElement('option'), { value: option.value, textContent: option.label })));
  box.onchange = async () => {
    const chosen = options.find(option => option.value === box.value);
    if (!chosen) return;
    if (!box.value) { renderHeadChoices(manifest.classifiers, manifest.classifiers); return; }
    const packManifest = await (await fetch(`/packs/${encodeURIComponent(box.value)}/manifest.json`)).json();
    renderHeadChoices(packManifest.classifiers, packManifest.classifiers);
  };
  renderHeadChoices(manifest.classifiers, manifest.classifiers);
}
function renderHeadChoices(all, selected) {
  const list = $('heads');
  list.replaceChildren(...all.map(id => {
    const item = Object.assign(document.createElement('label'), { className: 'check' });
    const input = Object.assign(document.createElement('input'), { type: 'checkbox', checked: selected.includes(id), value: id });
    input.onchange = update;
    item.append(input, Object.assign(document.createElement('span'), { textContent: LABELS[id] ?? id }));
    return item;
  }));
  function update() {
    const checked = [...list.querySelectorAll('input:checked')].length;
    $('apply').disabled = checked === 0;
    $('apply').textContent = `Load ${checked} of ${all.length} heads`;
  }
  update();
  const apply = async () => {
    const heads = [...list.querySelectorAll('input:checked')].map(input => input.value);
    if (!heads.length) { fail(new Error('Pick at least one head.')); return; }
    const pack = $('release').value;
    let next;
    if (!pack) {
      // The installed release: keep its prefix and only trim classifiers.
      next = { ...manifest, classifiers: heads };
    } else {
      const packManifest = await (await fetch(`/packs/${encodeURIComponent(pack)}/manifest.json`)).json();
      next = { ...packManifest, pack, classifiers: heads };
    }
    manifest = next;
    $('threshold').value = String(manifest.threshold);
    renderRows();
    $('provenance').textContent = `Release ${manifest.release}${manifest.pack ? ' (published pack)' : ''}, ${manifest.classifiers.length} of ${all.length} heads. ${manifest.training_data ?? ''}`;
    worker?.postMessage({ type: 'load', manifest, threshold: Number($('threshold').value) });
    status(`Loaded ${heads.length} head(s) from ${manifest.release}. Press Start to listen.`);
  };
  $('apply').onclick = apply;
}

function renderRows() {
  $('rows').replaceChildren(...manifest.classifiers.map((id) => {
    const row = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `<span class="score-name"></span><meter min="0" max="1" low="0.3" high="0.5" optimum="0" value="0"></meter><span class="score-value">0.00</span><span class="score-last">No detection yet</span>`;
    row.querySelector('.score-name').textContent = LABELS[id] ?? id;
    row.dataset.id = id;
    return row;
  }));
}

function onScores({ results, ms }) {
  for (const result of results) {
    frames++;
    msTotal += ms;
    if (day) for (const id of result.detections) { day.detection((result.frame + 1) * 1280, id, result.scores[id]); renderReview(); }
    recorder?.score({ f: result.frame, s: Object.fromEntries(Object.entries(result.scores).map(([k, v]) => [k, Math.round(v * 1000) / 1000])), d: result.detections });
    for (const [id, score] of Object.entries(result.scores)) {
      const row = $('rows').querySelector(`[data-id="${id}"]`);
      row.querySelector('meter').value = score;
      row.querySelector('.score-value').textContent = score.toFixed(2);
    }
    for (const id of result.detections) {
      const at = new Date().toLocaleTimeString();
      const score = result.scores[id];
      lastDetection = { id, atSample: (result.frame + 1) * 1280 };
      $('false').disabled = !recorder;
      $('rows').querySelector(`[data-id="${id}"] .score-last`).textContent = `Detected at ${at} (${score.toFixed(2)})`;
      log.unshift(`${at}  ${LABELS[id] ?? id}  score ${score.toFixed(2)}`);
      $('log').textContent = log.slice(0, 50).join('\n');
    }
  }
  $('timing').textContent = `${frames} decisions, ${(msTotal / frames).toFixed(1)} ms of CPU per 80 ms of audio`;
}

async function markNow(kind) {
  if (day && kind === 'missed') {
    day.missed();
    log.unshift(`${new Date().toLocaleTimeString()}  marked: missed it (last 5 s saved)`);
    $('log').textContent = log.slice(0, 50).join('\n');
    return;
  }
  if (day && kind === 'false-trigger') {
    const row = day.detections.find((d) => !d.verdict);
    if (row && (await day.mark('false-trigger', row)).ok) { row.verdict = 'false-trigger'; renderReview(); }
    return;
  }
  if (!recorder) return;
  const at = kind === 'false-trigger' ? lastDetection?.atSample : recorder.samples;
  if (at === undefined) return;
  const response = await recorder.mark(kind, at, kind === 'false-trigger' ? lastDetection.id : null);
  const time = new Date().toLocaleTimeString();
  log.unshift(response.ok ? `${time}  marked: ${kind === 'missed' ? 'missed it' : 'false trigger'}` : `${time}  mark not saved (${response.status})`);
  $('log').textContent = log.slice(0, 50).join('\n');
  if (kind === 'false-trigger') { lastDetection = null; $('false').disabled = true; }
}

async function start() {
  $('error').hidden = true;
  $('start').disabled = true;
  $('mode').disabled = true;
  try {
    status('Loading the detector…');
    if (!worker) {
      worker = new Worker('/detect-worker.mjs', { type: 'module' });
      await new Promise((resolve, reject) => {
        worker.onmessage = ({ data }) => data.type === 'ready' ? resolve() : data.type === 'error' ? reject(new Error(data.message)) : null;
        worker.postMessage({ type: 'load', manifest, threshold: Number($('threshold').value) });
      });
      worker.onmessage = ({ data }) => data.type === 'scores' ? onScores(data) : data.type === 'error' ? fail(data.message) : null;
    }
    worker.postMessage({ type: 'reset' });
    recorder = $('mode').value === 'full' ? new LiveRecorder() : null;
    day = $('mode').value === 'allday' ? new DayLog() : null;
    $('review-box').hidden = !day;
    lastDetection = null;
    mic = new LiveMicrophone();
    await mic.start((pcm) => {
      recorder?.push(pcm.slice());
      day?.push(pcm);
      worker.postMessage({ type: 'pcm', pcm }, [pcm.buffer]);
    });
    $('stop').disabled = false;
    $('missed').disabled = !recorder && !day;
    renderRecording();
    status(day ? 'Listening all day. Work normally; press “Missed it” after a wake word that did not fire, and mark detections Right or Wrong below.'
      : recorder ? 'Listening and recording. Talk normally; press “Missed it” right after a wake word that did not fire.'
      : 'Listening (nothing saved). Say your wake phrase.');
  } catch (error) {
    mic?.stop();
    mic = null;
    recorder = null;
    day = null;
    $('start').disabled = false;
    $('mode').disabled = false;
    status('Stopped.');
    fail(error);
  }
}

async function stop() {
  mic?.stop();
  mic = null;
  $('stop').disabled = true;
  $('missed').disabled = true;
  $('false').disabled = true;
  if (day) {
    status('Saving the last snippets…');
    await new Promise((resolve) => setTimeout(resolve, 300));
    await day.close();
    renderReview();
    status(`Stopped after ${(day.samples / 16000 / 60).toFixed(0)} min. ${day.saved} snippets saved privately on this host.`);
    day = null;
  } else if (recorder) {
    status('Saving the last part of the recording…');
    await new Promise((resolve) => setTimeout(resolve, 300)); // let the worker post its last scores
    await recorder.flush();
    renderRecording();
    status(`Stopped. ${(recorder.saved / 16000 / 60).toFixed(1)} min saved privately on this host for transcription.`);
    recorder = null;
  } else {
    status('Stopped. The microphone is closed.');
  }
  $('start').disabled = false;
  $('mode').disabled = false;
}

addEventListener('DOMContentLoaded', async () => {
  $('start').addEventListener('click', start);
  $('stop').addEventListener('click', stop);
  $('missed').addEventListener('click', () => markNow('missed'));
  $('false').addEventListener('click', () => markNow('false-trigger'));
  $('mode').addEventListener('change', renderRecording);
  $('speak').addEventListener('click', async () => {
    // Self-trigger test on this device: the assistant's own voice must not wake it.
    const audio = new Audio('/spike/tts-wake.wav');
    const playStarted = performance.now();
    worker?.postMessage({ type: 'playback', active: true });
    $('speak').disabled = true;
    audio.onended = () => { worker?.postMessage({ type: 'playback', active: false }); $('speak').disabled = false; if (day) day.pausedSeconds += (performance.now() - playStarted) / 1000 + 0.35; };
    audio.onerror = () => { worker?.postMessage({ type: 'playback', active: false }); $('speak').disabled = false; fail('The test phrase could not be played.'); };
    try { await audio.play(); log.unshift(`${new Date().toLocaleTimeString()}  played an assistant phrase (half-duplex, microphone ignored)`); $('log').textContent = log.slice(0, 50).join('\n'); }
    catch (error) { worker?.postMessage({ type: 'playback', active: false }); $('speak').disabled = false; fail(error); }
  });
  $('threshold').addEventListener('change', () => worker?.postMessage({ type: 'threshold', threshold: Number($('threshold').value) }));
  try {
    manifest = await (await fetch('/models/manifest.json')).json();
    $('threshold').value = String(manifest.threshold);
    renderPicker();
    renderRows();
    renderRecording();
    // training_data is the release's own account of what it learned from; older releases only have the description
    $('provenance').textContent = `Release ${manifest.release}. ${manifest.training_data ?? manifest.description}`;
    $('start').disabled = false;
    status('Press Start to listen.');
  } catch (error) {
    status('No detector is installed on this page yet.');
    fail(error);
  }
});

window.wakeDetect = {
  recorder: () => recorder && { sessionId: recorder.sessionId, samples: recorder.samples, saved: recorder.saved, failed: recorder.failed },
  day: () => day && { sessionId: day.sessionId, samples: day.samples, saved: day.saved, failed: day.failed, detections: day.detections.length },
};
