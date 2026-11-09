// Pure display state; scores and byte counts are observations, never action authority.
const NAMES = { oqto: 'Oqto', elissa: 'Elissa', pi: 'Pi', opencode: 'OpenCode', hex: 'Hex', governor: 'Governor', colonel: 'Colonel', snake: 'Snake', 'mei-ling': 'Mei-Ling', meryl: 'Meryl', otacon: 'Otacon', miller: 'Miller', naomi: 'Naomi', ginee: 'Ginee' };
export function phraseLabel(id) {
  const aliases = { hey_r2: 'Hey Oqto · shipped', oqto_r2: 'Oqto · shipped', hey_new: 'Hey Oqto · new', oqto_new: 'Oqto · new' };
  if (Object.hasOwn(aliases, id)) return aliases[id];
  const addressed = id.startsWith('hey_');
  const word = addressed ? id.slice(4) : id;
  const label = Object.hasOwn(NAMES, word) ? NAMES[word] : word.replaceAll('_', ' ').replace(/\b[a-z]/g, c => c.toUpperCase());
  return addressed ? `Hey ${label}` : label;
}
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return 'unknown size';
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}
export function fitPage(width, height, count, fontSize = 16) {
  const rowHeight = (width < 740 ? 3.25 : 2.75) * fontSize;
  const maxColumns = Math.max(1, Math.min(4, Math.floor((width + 24) / (24 * fontSize + 24))));
  const rows = Math.max(1, Math.floor((height - 24) / rowHeight));
  const pageSize = rows * maxColumns;
  const columns = Math.max(1, Math.min(maxColumns, Math.ceil(Math.min(count, pageSize) / rows)));
  return { rowHeight, rows, columns, pageSize, pages: Math.max(1, Math.ceil(count / pageSize)) };
}
export function selectedModelBytes(releases) {
  if (!releases.length) return 0;
  const shared = releases[0].shared_bytes;
  if (!Number.isFinite(shared)) return null;
  let total = shared;
  for (const release of releases) for (const id of release.classifiers) {
    const bytes = release.head_bytes?.[id];
    if (!Number.isFinite(bytes)) return null;
    total += bytes;
  }
  return total; // The worker shares one compatible front end; each classifier is instantiated.
}
export class HeadBoard {
  heads = new Map();
  preferences = new Map();
  timings = [];
  recent = [];
  frames = 0;
  processed = 0;
  sent = 0;
  inputPeak = null;
  maxInputPeak = 0;
  setReleases(releases) {
    this.heads = new Map();
    for (const release of releases) for (const id of release.classifiers) {
      const key = `${release.name}::${id}`;
      const threshold = this.preferences.get(key) ?? (Number.isFinite(release.threshold) ? release.threshold : 0.5);
      this.heads.set(key, { key, id, release: release.name, label: phraseLabel(id), bytes: release.head_bytes?.[id], threshold, score: null, peak: null, above: false });
    }
    this.reset();
  }
  reset() {
    this.timings = []; this.recent = []; this.frames = 0; this.processed = 0; this.sent = 0; this.inputPeak = null; this.maxInputPeak = 0;
    for (const head of this.heads.values()) { head.score = null; head.peak = null; head.above = false; }
  }
  threshold(key, value) {
    const head = this.heads.get(key);
    if (!head || !Number.isFinite(value) || value < 0 || value > 1) return false;
    head.threshold = value;
    this.preferences.set(key, value);
    head.above = head.score !== null && head.score >= value;
    return true;
  }
  observe({ results = [], ms, samples, processedSamples }) {
    if (Number.isFinite(ms) && ms >= 0 && samples > 0) {
      this.timings.push(ms * 1280 / samples);
      if (this.timings.length > 120) this.timings.shift();
    }
    if (Number.isFinite(processedSamples)) this.processed = processedSamples;
    for (const result of results) {
      this.frames++;
      for (const [key, value] of Object.entries(result.scores)) {
        const head = this.heads.get(key);
        if (!head || !Number.isFinite(value)) continue;
        const score = Math.max(0, Math.min(1, value));
        const above = score >= head.threshold;
        if (above && !head.above) this.recent.unshift({ key, label: head.label, release: head.release, score, time: (result.frame + 1) * 0.08 });
        head.score = score; head.peak = Math.max(head.peak ?? 0, score); head.above = above;
      }
      this.recent = this.recent.slice(0, 8);
    }
  }
  stats() {
    const n = this.timings.length;
    const mean = n ? this.timings.reduce((a, b) => a + b, 0) / n : null;
    const sorted = [...this.timings].sort((a, b) => a - b);
    return { mean, p95: n ? sorted[Math.ceil(n * .95) - 1] : null,
      realtimePercent: mean === null ? null : mean / 80 * 100,
      backlogMs: Math.max(0, this.sent - this.processed) / 16,
      inputDb: this.inputPeak === null ? null : 20 * Math.log10(Math.max(1e-6, this.inputPeak)) };
  }
}
