// Review page, two modes chosen by the clip manifest:
//   round mode  - one vote per (language, round); used for pronunciation decisions
//   label mode  - per-clip verdicts (genuine/false/lookalike/unsure) plus an editable transcript
// Everything saves as you go: labels on click, transcripts debounced while typing.
const LANGS = { de: 'German', en: 'English', es: 'Spanish' };
const ROUNDS = { 1: 'Round 1 — measured spellings', 2: 'Round 2 — candidate spellings' };
const LABEL_KINDS = ['genuine', 'false', 'lookalike', 'unsure'];
const SPELLING_HINT = 'Spell the transcript the way a naive listener would: German "Okto", English "Octo", ' +
  'Spanish "Octo". Write "Oqto" only if the speaker spells it out — the product name is mapped in the dictionary layer, not by the ASR.';

const app = document.getElementById('app');
const status = document.createElement('p');
status.id = 'status';
let manifest = [];
let votes = {};
let labels = {};
let edits = {};

const el = (tag, props = {}) => Object.assign(document.createElement(tag), props);
const getJson = async (url, fallback) => {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : fallback;
  } catch {
    return fallback;
  }
};
const postJson = (url, body) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// ---------------------------------------------------------------- round mode
function bySpelling(lang, round) {
  const groups = new Map();
  for (const clip of manifest.filter((c) => c.lang === lang && (c.round ?? 1) === round)) {
    if (!groups.has(clip.spelling)) groups.set(clip.spelling, []);
    groups.get(clip.spelling).push(clip);
  }
  return groups;
}

function renderRounds() {
  app.replaceChildren();
  for (const [lang, label] of Object.entries(LANGS)) {
    for (const round of [1, 2]) {
      if (!manifest.some((c) => c.lang === lang && (c.round ?? 1) === round)) continue;
      app.append(el('h2', { textContent: `${label} · ${ROUNDS[round] ?? `Round ${round}`}` }));
      for (const [spelling, clips] of bySpelling(lang, round)) {
        const box = el('section', { className: 'spell' });
        const heard = [...new Set(clips.map((c) => c.heard))].join(' · ');
        const head = el('h3');
        head.append(el('strong', { textContent: spelling }), el('span', { className: 'heard', textContent: `heard: ${heard}` }));
        box.append(head);
        for (const clip of clips) {
          const row = el('div', { className: 'clip' });
          const audio = new Audio(clip.clip);
          const play = el('button', { textContent: 'Play', onclick: () => { audio.currentTime = 0; audio.play(); } });
          row.append(play, el('span', { className: 'text', textContent: clip.text }), el('span', { className: 'heard', textContent: clip.heard ?? '' }));
          box.append(row);
        }
        const key = `${lang}-r${round}`;
        const vote = el('div', { className: 'vote' });
        const radio = el('input', { type: 'radio', name: `vote-${key}`, value: spelling });
        radio.checked = votes[key] === spelling;
        radio.onchange = () => { votes[key] = spelling; };
        vote.append(el('label', {}).appendChild(radio) ? radio.parentElement : radio);
        const labelEl = el('label');
        labelEl.append(radio, document.createTextNode('sounds right'));
        box.append(el('div', { className: 'vote' }).appendChild(labelEl) ? box.lastElementChild : box);
        app.append(box);
      }
      const bar = el('div', { className: 'vote' });
      const save = el('button', { textContent: `Save my choice (${label}, round ${round})` });
      save.onclick = async () => {
        const key = `${lang}-r${round}`;
        if (!votes[key]) { status.textContent = 'pick a spelling first'; return; }
        const res = await postJson('/api/vote', { lang: key, spelling: votes[key] });
        status.textContent = res.ok ? `saved ${label} round ${round}: ${votes[key]}` : `save failed (${res.status})`;
      };
      bar.append(save, status);
      app.append(bar);
    }
  }
}

// ---------------------------------------------------------------- label mode
function labelTally() {
  const counts = Object.fromEntries(LABEL_KINDS.map((k) => [k, 0]));
  for (const clip of manifest) {
    const label = labels[clip.id]?.label;
    if (label) counts[label] += 1;
  }
  const marked = manifest.filter((c) => labels[c.id]?.label).length;
  return `marked ${marked} of ${manifest.length} · ` +
    LABEL_KINDS.filter((k) => counts[k]).map((k) => `${k} ${counts[k]}`).join(' · ');
}

function transcriptEditor(clip) {
  const wrap = el('div', { className: 'vote' });
  const input = el('input', { type: 'text', value: edits[clip.id]?.text ?? clip.heard ?? '', placeholder: 'what was actually said' });
  Object.assign(input.style, { flex: '1', minWidth: '18rem', padding: '6px 8px', background: 'var(--surface)',
    color: 'var(--fg)', border: '1px solid var(--rule)', borderRadius: '6px' });
  const mark = el('span', { textContent: edits[clip.id] ? 'corrected' : '' });
  let timer;
  const save = async () => {
    const res = await postJson('/api/transcript', { id: clip.id, text: input.value });
    if (!res.ok) { status.textContent = `transcript save failed (${res.status})`; return; }
    if (input.value) edits[clip.id] = { text: input.value }; else delete edits[clip.id];
    mark.textContent = input.value ? 'corrected · saved' : '';
  };
  input.oninput = () => { mark.textContent = 'editing…'; clearTimeout(timer); timer = setTimeout(save, 600); };
  input.onblur = () => { clearTimeout(timer); save(); };
  const revert = el('button', { textContent: 'ASR text', onclick: () => { input.value = clip.heard ?? ''; save(); } });
  wrap.append(input, revert, mark);
  return wrap;
}

function renderLabels() {
  app.replaceChildren();
  const tally = el('p', { className: 'lede', textContent: labelTally() });
  const refresh = () => { tally.textContent = labelTally(); };
  app.append(tally, el('p', { className: 'lede', textContent: SPELLING_HINT }));

  for (const clip of manifest) {
    const box = el('section', { className: 'spell' });
    const head = el('h3');
    head.append(
      el('strong', { textContent: `${clip.session} · ${clip.file}` }),
      el('span', { className: 'heard', textContent: `detector ${Number(clip.detector_score ?? 0).toFixed(2)}` }),
    );
    box.append(head);

    const row = el('div', { className: 'clip' });
    const audio = new Audio(clip.clip);
    row.append(el('button', { textContent: 'Play', onclick: () => { audio.currentTime = 0; audio.play(); } }));
    box.append(row);

    const timed = (clip.words ?? []).map((w) => `${w.w} ${w.t0.toFixed(2)}-${w.t1.toFixed(2)}`).join('  ');
    const details = [(clip.heard ? `transcript: ${clip.heard}` : 'transcript: (nothing recognised)')];
    if (clip.at_offset !== undefined) {
      details.push(`detected at ${Number(clip.at_offset).toFixed(2)} s` +
        (clip.fired_on ? `, nearest word: ${clip.fired_on} (${clip.distance} s away)` : ', no word at that moment'));
    }
    if (timed) details.push(`words: ${timed}`);
    box.append(el('p', { className: 'heard', textContent: details.join('\n') }));

    box.append(transcriptEditor(clip));

    const bar = el('div', { className: 'vote' });
    for (const kind of clip.labels ?? LABEL_KINDS) {
      const button = el('button', { textContent: kind });
      button.style.opacity = labels[clip.id]?.label === kind ? '1' : '0.6';
      if (clip.proposed === kind) button.style.borderColor = 'var(--primary)';
      button.onclick = async () => {
        const res = await postJson('/api/label', { id: clip.id, label: kind });
        if (!res.ok) { status.textContent = `save failed (${res.status})`; return; }
        labels[clip.id] = { label: kind };
        for (const sibling of bar.children) sibling.style.opacity = '0.6';
        button.style.opacity = '1';
        refresh();
      };
      bar.append(button);
    }
    box.append(bar);
    app.append(box);
  }
  app.append(status);
}

// ---------------------------------------------------------------- boot
manifest = await getJson('/clips.json', []);
votes = await getJson('/votes.json', {});
labels = await getJson('/labels.json', {});
edits = await getJson('/transcripts.json', {});
const labelMode = Array.isArray(manifest) && manifest.length > 0 && Array.isArray(manifest[0].labels);
if (labelMode) renderLabels(); else renderRounds();
