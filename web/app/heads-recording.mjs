// Opt-in continuous diagnostics: bounded audio RAM, lossless head logs on the private host.
export const RECORD_RATE = 16000;
export const RECORD_CHUNK = 1280 * 100; // 8 s, exactly 100 detector frames
export const RECORD_LIMIT = RECORD_RATE * 60 * 120;
export const MAX_RECORDED_HEADS = 512;

export class HeadRecording {
  constructor({ heads, wav, send = (...args) => fetch(...args), onChange = () => {}, intent = 'negative-only' }) {
    if (heads.length > MAX_RECORDED_HEADS) throw new Error(`Record at most ${MAX_RECORDED_HEADS} heads.`);
    this.sessionId = crypto.randomUUID(); this.startedAt = new Date().toISOString();
    this.heads = heads; this.wav = wav; this.send = send; this.onChange = onChange; this.intent = intent;
    this.parts = []; this.frames = []; this.events = []; this.above = new Map(); this.previous = new Map();
    this.samples = 0; this.processed = 0; this.flushed = 0; this.saved = 0; this.seq = 0;
    this.queue = Promise.resolve(); this.error = null; this.failed = []; this.finished = false;
  }
  push(pcm) {
    if (this.error) throw new Error(this.error);
    if (this.samples + pcm.length > RECORD_LIMIT) throw new Error('Recording reached two hours. Start a new session.');
    if (this.samples - this.saved > RECORD_CHUNK * 4) throw new Error('Saving cannot keep up. Recording stopped before dropping audio.');
    if (!this.samples) this.startedAt = new Date().toISOString();
    this.parts.push(pcm.slice()); this.samples += pcm.length;
  }
  threshold(key, value) { this.above.set(key, (this.previous.get(key) ?? -1) >= value); }
  observe({ results, processedSamples }, heads) {
    this.processed = processedSamples;
    for (const result of results) {
      const s = [], t = [], d = [];
      this.heads.forEach((key, i) => {
        const score = result.scores[key], threshold = heads.get(key).threshold;
        s.push(score); t.push(threshold);
        if (score >= threshold && !this.above.get(key)) {
          d.push(i); this.events.push({ key, atSample: (result.frame + 1) * 1280, score, threshold, verdict: null });
        }
        this.above.set(key, score >= threshold); this.previous.set(key, score);
      });
      this.frames.push({ f: result.frame, s, t, d });
      // Review shows recent crossings; every crossing remains in the on-disk frame logs.
      if (this.events.length > 500) this.events = this.events.slice(-500);
    }
    while (this.flushed + RECORD_CHUNK <= Math.min(this.processed, this.samples)) this.flush(this.flushed + RECORD_CHUNK);
  }
  flush(end) {
    if (end <= this.flushed) return;
    const startSample = this.flushed, pcm = new Float32Array(end - startSample);
    let used = 0;
    while (used < pcm.length) {
      const part = this.parts.shift(), n = Math.min(part.length, pcm.length - used);
      pcm.set(part.subarray(0, n), used); used += n;
      if (n < part.length) this.parts.unshift(part.subarray(n));
    }
    const headFrames = this.frames.filter(row => (row.f + 1) * 1280 <= end);
    this.frames = this.frames.filter(row => (row.f + 1) * 1280 > end);
    const meta = { sessionId: this.sessionId, seq: this.seq++, startSample, release: 'head-explorer', threshold: null,
      scores: [], mode: 'heads', intent: this.intent, startedAt: this.startedAt, heads: this.heads,
      modelHashes: this.modelHashes, headFrames,
      processedSamples: Math.min(this.processed, end) };
    this.flushed = end;
    const job = { audio: this.wav(pcm), meta, samples: pcm.length };
    this.queue = this.queue.then(() => this.upload(job));
  }
  async upload(job) {
    try {
      const body = new FormData(); body.set('audio', job.audio, 'chunk.wav'); body.set('meta', JSON.stringify(job.meta));
      const response = await this.send('/api/live/chunk', { method: 'POST', credentials: 'same-origin', body, signal: AbortSignal.timeout(15000) });
      // A lost response may leave an identical immutable chunk saved; UUID + seq identify this upload.
      const duplicate = response.status === 409 && (await response.text()) === 'Chunk already saved';
      if (!response.ok && !duplicate) throw new Error(`Recording save failed (${response.status}). Stop and retry saving.`);
      this.saved += job.samples;
    } catch (error) { this.error = error.message; this.failed.push(job); }
    this.onChange();
  }
  async finish() { this.flush(this.samples); await this.queue; this.finished = true; this.onChange(); }
  async retry() {
    const failed = this.failed.splice(0); this.error = null;
    for (const job of failed) await this.upload(job);
  }
  async mark(event) {
    if (!event) return;
    const response = await this.send('/api/live/mark', { method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(15000),
      body: JSON.stringify({ sessionId: this.sessionId, kind: 'false-trigger', atSample: event.atSample, detection: event.key }) });
    if (!response.ok) throw new Error(`False-trigger mark could not be saved (${response.status}).`);
    event.verdict = 'false-trigger'; this.onChange();
  }
}
