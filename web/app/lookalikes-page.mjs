// Look-alike editor: every word's near-miss phrases, remove / restore / add, delete the takes of
// removed or unlisted phrases. The server owns validation and the private overrides file; this
// page only renders its state and sends one edit at a time with the revision it last saw.

const $ = id => document.getElementById(id);
const LANG_NAMES = { de: 'German', en: 'English' };
let state = null;
let language = 'both';
let query = '';
let pendingDelete = null;   // `${word}|${language}|${line}` awaiting its second click
let undo = null;            // { label, edit }
let busy = false;

const say = (text, action = null) => {
  const root = $('status');
  root.replaceChildren(Object.assign(document.createElement('span'), { textContent: text }));
  if (action) {
    const button = Object.assign(document.createElement('button'), { type: 'button', textContent: action.label });
    button.onclick = action.run;
    root.append(button);
  }
};
const fail = message => { $('error').textContent = message; $('error').hidden = false; };
const clearError = () => { $('error').hidden = true; $('error').textContent = ''; };
const itemKey = (word, lang, line) => `${word}|${lang}|${line}`;
const langs = () => (language === 'both' ? ['de', 'en'] : [language]);
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

async function load() {
  const response = await fetch('/api/lookalikes', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load the list (${response.status})`);
  state = await response.json();
  render();
}

async function send(edit, { focus } = {}) {
  if (busy) return null;
  busy = true; clearError();
  try {
    const response = await fetch('/api/lookalikes', {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...edit, revision: state.revision }),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 409 && body.state) { state = body.state; render(); fail(body.error); return null; }
    if (!response.ok) { fail(body.error ?? `Could not save (${response.status})`); return null; }
    state = body;
    render(focus);
    return body;
  } catch (error) {
    fail(`Could not save: ${error.message}`);
    return null;
  } finally { busy = false; }
}

async function remove(word, lang, item) {
  const done = await send({ op: 'remove', word: word.name, language: lang, line: item.line }, { focus: itemKey(word.name, lang, item.line) });
  if (!done) return;
  const back = item.source === 'owner' ? { op: 'add', word: word.name, language: lang, line: item.line }
    : { op: 'restore', word: word.name, language: lang, line: item.line };
  undo = { edit: back };
  say(`Removed “${item.line}” from ${word.display} (${lang.toUpperCase()}).${item.recorded ? ` Its ${plural(item.recorded, 'take')} stay until you delete them.` : ''}`,
    { label: 'Undo', run: () => runUndo() });
}
async function restore(word, lang, item) {
  if (await send({ op: 'restore', word: word.name, language: lang, line: item.line }, { focus: itemKey(word.name, lang, item.line) })) {
    say(`Restored “${item.line}” to ${word.display} (${lang.toUpperCase()}).`);
  }
}
async function add(word, lang, input) {
  const line = input.value.trim();
  if (!line) { input.focus(); return; }
  if (await send({ op: 'add', word: word.name, language: lang, line }, { focus: `add|${word.name}|${lang}` })) {
    undo = { edit: { op: 'remove', word: word.name, language: lang, line } };
    say(`Added “${line}” to ${word.display} (${lang.toUpperCase()}).`, { label: 'Undo', run: () => runUndo() });
  }
}
async function runUndo() {
  if (!undo) return;
  const { edit } = undo; undo = null;
  if (await send(edit, { focus: itemKey(edit.word, edit.language, edit.line) })) say('Undone.');
}
async function deleteTakes(word, lang, line, count) {
  const done = await send({ op: 'delete-takes', word: word.name, language: lang, line });
  pendingDelete = null;
  if (done) { render(); say(`Deleted ${plural(done.deleted ?? count, 'take')} of “${line}” (${word.display}, ${lang.toUpperCase()}).`); }
}

function matches(word, lang, line) {
  if (!query) return true;
  return word.display.toLowerCase().includes(query) || word.name.includes(query) || line.toLowerCase().includes(query);
}

// Every row is the same grid: phrase (with its details underneath), then two equal action
// slots. The right slot is the row's main action, the middle one the secondary action, and an
// empty slot keeps its place so buttons line up down the page.
function row(word, lang, item, kind) {
  const k = itemKey(word.name, lang, item.line);
  const li = document.createElement('li');
  li.className = 'lk-item';
  li.dataset.key = k;
  li.dataset.state = kind;
  const label = document.createElement('span'); label.className = 'lk-label';
  const meta = [];
  if (kind === 'orphan') meta.push('not in the list');
  if (kind === 'removed') meta.push('removed');
  if (item.source === 'owner') meta.push('added by you');
  if (item.recorded) meta.push(plural(item.recorded, 'take'));
  label.append(Object.assign(document.createElement('span'), { className: 'lk-line', textContent: item.line }));
  if (meta.length) label.append(Object.assign(document.createElement('span'), { className: 'lk-meta', textContent: meta.join(' · ') }));
  const button = (text, run, cls, aria) => {
    const b = Object.assign(document.createElement('button'), { type: 'button', textContent: text, className: `lk-action ${cls}` });
    b.setAttribute('aria-label', `${aria} “${item.line}”, ${word.display} ${LANG_NAMES[lang]}`);
    b.onclick = run; return b;
  };
  const slot = () => Object.assign(document.createElement('span'), { className: 'lk-slot', ariaHidden: 'true' });
  const takes = item.recorded ? plural(item.recorded, 'take') : '';
  const deletable = (kind === 'removed' || kind === 'orphan') && item.recorded;
  let middle = slot(), right = slot();
  if (deletable && pendingDelete === k) {
    middle = button('Confirm', () => deleteTakes(word, lang, item.line, item.recorded), 'confirm', `Confirm: delete ${takes} of`);
    right = button('Cancel', () => { pendingDelete = null; render(k); }, 'quiet', 'Keep the takes of');
  } else {
    const askDelete = () => button('Delete takes', () => { pendingDelete = k; render(k); }, 'quiet warn', `Delete ${takes} of`);
    if (kind === 'active') right = button('Remove', () => remove(word, lang, item), 'quiet', 'Remove');
    if (kind === 'removed') right = button('Restore', () => restore(word, lang, item), 'quiet', 'Restore');
    if (kind === 'removed' && deletable) middle = askDelete();
    if (kind === 'orphan') right = askDelete();
  }
  // An empty middle slot lends its width to the text; the right column never moves.
  if (middle.classList.contains('lk-slot')) { label.classList.add('lk-label-wide'); li.append(label, right); }
  else li.append(label, middle, right);
  return li;
}

function renderRail() {
  const nodes = state.words.map(word => {
    let active = 0, cleanup = 0;
    for (const lang of langs()) {
      const lists = word.languages[lang];
      if (!lists) continue;
      active += lists.items.filter(item => !item.removed).length;
      cleanup += lists.orphans.length + lists.items.filter(item => item.removed && item.recorded).length;
    }
    const li = document.createElement('li');
    const a = Object.assign(document.createElement('a'), { className: 'rail-word', href: `#w-${word.name}` });
    a.append(Object.assign(document.createElement('span'), { className: 'rail-name', textContent: word.display }),
      Object.assign(document.createElement('span'), { className: `rail-counts${cleanup ? ' rail-cleanup' : ''}`, textContent: cleanup ? `${active} · ${cleanup} to clean` : String(active) }));
    a.setAttribute('aria-label', `${word.display}: ${plural(active, 'look-alike')}${cleanup ? `, ${cleanup} with takes to clean up` : ''}`);
    li.append(a);
    return li;
  });
  $('rail').replaceChildren(...nodes);
  $('rail-total').textContent = `${state.counts.active} active`;
}

function render(focusKey = null) {
  if (!state) return;
  const { counts } = state;
  $('summary').textContent = `${plural(counts.active, 'look-alike')} · ${counts.removed} removed · ${counts.added} added by you`
    + (counts.cleanup ? ` · ${plural(counts.cleanup, 'phrase')} with takes to clean up` : '');
  renderRail();
  const sections = [];
  for (const word of state.words) {
    const blocks = [];
    for (const lang of langs()) {
      const lists = word.languages[lang];
      if (!lists) continue;
      const active = lists.items.filter(item => !item.removed && matches(word, lang, item.line));
      const removed = lists.items.filter(item => item.removed && matches(word, lang, item.line));
      const orphans = lists.orphans.filter(item => matches(word, lang, item.line));
      if (query && !active.length && !removed.length && !orphans.length) continue;
      const block = document.createElement('div'); block.className = 'lk-lang';
      block.append(Object.assign(document.createElement('h3'), { textContent: LANG_NAMES[lang] }));
      const wake = document.createElement('p'); wake.className = 'lk-wake';
      wake.append(Object.assign(document.createElement('span'), { textContent: 'Wake: ' }),
        Object.assign(document.createElement('strong'), { textContent: lists.wake.join(' · ') }));
      if (word.accept.length) wake.append(Object.assign(document.createElement('span'), { className: 'lk-accept', textContent: ` · same sound: ${word.accept.join(', ')}` }));
      block.append(wake);
      const ul = document.createElement('ul'); ul.className = 'lk-items';
      for (const item of active) ul.append(row(word, lang, item, 'active'));
      for (const item of orphans) ul.append(row(word, lang, { ...item, source: 'recorded' }, 'orphan'));
      for (const item of removed) ul.append(row(word, lang, item, 'removed'));
      if (!ul.children.length) ul.append(Object.assign(document.createElement('li'), { className: 'lk-empty', textContent: 'No look-alikes yet.' }));
      block.append(ul);
      const form = document.createElement('form'); form.className = 'lk-add';
      const input = Object.assign(document.createElement('input'), { type: 'text', maxLength: 60, placeholder: 'Add a look-alike', autocomplete: 'off', spellcheck: false });
      input.dataset.key = `add|${word.name}|${lang}`;
      input.setAttribute('aria-label', `Add a ${LANG_NAMES[lang]} look-alike for ${word.display}`);
      const submit = Object.assign(document.createElement('button'), { type: 'submit', textContent: 'Add', className: 'lk-action' });
      form.append(input, submit);
      form.onsubmit = event => { event.preventDefault(); void add(word, lang, input); };
      block.append(form);
      blocks.push(block);
    }
    if (!blocks.length) continue;
    const section = document.createElement('section'); section.className = 'lk-word'; section.id = `w-${word.name}`;
    section.setAttribute('aria-labelledby', `h-${word.name}`);
    section.append(Object.assign(document.createElement('h2'), { id: `h-${word.name}`, textContent: word.display }));
    const grid = document.createElement('div'); grid.className = 'lk-langs'; grid.append(...blocks);
    section.append(grid);
    sections.push(section);
  }
  $('list').replaceChildren(...(sections.length ? sections : [Object.assign(document.createElement('p'), { className: 'hint', textContent: `Nothing matches “${query}”.` })]));
  if (focusKey) {
    // Keep the keyboard where the edit happened: same row, else its list's add field.
    const target = document.querySelector(`[data-key="${CSS.escape(focusKey)}"] button, input[data-key="${CSS.escape(focusKey)}"]`)
      ?? document.querySelector(`input[data-key="${CSS.escape(`add|${focusKey.split('|').slice(0, 2).join('|')}`)}"]`);
    target?.focus({ preventScroll: false });
  }
}

$('search').oninput = () => { query = $('search').value.trim().toLowerCase(); pendingDelete = null; render(); };
for (const button of $('language').querySelectorAll('button')) {
  button.onclick = () => {
    language = button.dataset.value;
    for (const sibling of $('language').querySelectorAll('button')) sibling.setAttribute('aria-pressed', String(sibling === button));
    pendingDelete = null; render();
  };
}
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && pendingDelete) { const k = pendingDelete; pendingDelete = null; render(k); }
});

load().then(() => {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  target?.scrollIntoView({ block: 'start' });
}).catch(error => fail(error.message));
window.lookalikeEditor = { state: () => state };
