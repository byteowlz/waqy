import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

// Private owner-recording store for wake-phrase candidates. Server-generated IDs;
// the client never names files or paths. Recordings never enter git.
export const DEFAULT_ROOT = path.join(process.env.HOME ?? '/', '.local/share/foxline/wake-collect-url-1');
const MAX_TAKE_SECONDS = 4;
const MAX_WAV_BYTES = 44 + 16000 * 2 * (MAX_TAKE_SECONDS + 0.5);
const ID = /^\d{8}T\d{6}Z-[0-9a-f]{8}$/;
const SESSION = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const clean = (value, max) => typeof value === 'string' ? value.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '').slice(0, max) : '';
const now = () => new Date().toISOString();

export class RecordingError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

// Only the exact PCM16 mono 16 kHz WAV the panel's capture produces.
export function parseWav(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = offset => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  if (bytes.byteLength < 46) throw new RecordingError(413, 'Recording too small');
  const ok = tag(0) === 'RIFF' && view.getUint32(4, true) === bytes.byteLength - 8 && tag(8) === 'WAVE' && tag(12) === 'fmt '
    && view.getUint32(16, true) === 16 && view.getUint16(20, true) === 1 && view.getUint16(22, true) === 1
    && view.getUint32(24, true) === 16000 && view.getUint32(28, true) === 32000 && view.getUint16(32, true) === 2
    && view.getUint16(34, true) === 16 && tag(36) === 'data' && view.getUint32(40, true) === bytes.byteLength - 44
    && (bytes.byteLength - 44) % 2 === 0;
  if (!ok) throw new RecordingError(415, 'Expected PCM16 mono 16 kHz WAV');
  const samples = (bytes.byteLength - 44) / 2;
  if (bytes.byteLength > MAX_WAV_BYTES || samples > 16000 * (MAX_TAKE_SECONDS + 0.5)) throw new RecordingError(413, 'Wake take must be at most 4 seconds');
  // A wake reference must contain something audible; all-zero takes are refused.
  let peak = 0;
  for (let i = 44; i < bytes.byteLength; i += 2) peak = Math.max(peak, Math.abs(view.getInt16(i, true)));
  if (peak === 0) throw new RecordingError(415, 'Recording is silent');
  return { samples, durationSeconds: samples / 16000, peakPcm: peak };
}

// Multi-mic recording multiplies takes (five devices = five takes per phrase), so the count cap is
// generous by default and both caps can be set per host.
export const DEFAULT_MAX_COUNT = Number(process.env.RECORDINGS_MAX_COUNT ?? 50_000);
export const DEFAULT_MAX_BYTES = Number(process.env.RECORDINGS_MAX_BYTES ?? 8 * 1024 ** 3);

export function createRecordingStore({ root = DEFAULT_ROOT, maxCount = DEFAULT_MAX_COUNT, maxBytes = DEFAULT_MAX_BYTES } = {}) {
  // A bad env value must not turn the cap into NaN (which never refuses and never reports).
  if (!Number.isFinite(maxCount) || maxCount <= 0) maxCount = DEFAULT_MAX_COUNT;
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) maxBytes = DEFAULT_MAX_BYTES;
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  fs.chmodSync(root, 0o700);
  const dir = id => {
    if (!ID.test(id)) throw new RecordingError(404, 'Unknown recording');
    const target = path.join(root, id);
    if (path.dirname(target) !== root || !fs.existsSync(path.join(target, 'meta.json'))) throw new RecordingError(404, 'Unknown recording');
    return target;
  };
  // A root removed out from under a running instance must not turn into a 500 on
  // every take: recreate it (private mode) instead of crashing the store.
  const ensureRoot = () => { if (!fs.existsSync(root)) fs.mkdirSync(root, { recursive: true, mode: 0o700 }); };
  const usage = () => {
    ensureRoot();
    let count = 0, bytes = 0;
    for (const entry of fs.readdirSync(root)) {
      if (!ID.test(entry)) continue;
      count++;
      for (const file of fs.readdirSync(path.join(root, entry))) bytes += fs.statSync(path.join(root, entry, file)).size;
    }
    return { count, bytes };
  };
  const write = (file, text) => fs.writeFileSync(file, text, { mode: 0o600, flag: 'wx' });

  return {
    root,
    create(wav, meta = {}, context = {}) {
      ensureRoot();
      const { samples, durationSeconds, peakPcm } = parseWav(wav);
      const current = usage();
      if (current.count >= maxCount || current.bytes + wav.byteLength > maxBytes) throw new RecordingError(507, 'Recording store is full');
      if (!SESSION.test(meta.sessionId ?? '')) throw new RecordingError(400, 'sessionId must be a page-session UUID');
      const group = clean(meta.group, 80).trim();
      const phrase = clean(meta.phrase, 200).trim();
      const spoken = clean(meta.spoken, 200).trim();
      const language = ['de', 'en'].includes(meta.language) ? meta.language : null;
      const label = ['positive', 'negative', 'silence'].includes(meta.label) ? meta.label : null;
      const take = Number.parseInt(meta.take, 10);
      // Older pages sent no style: those takes were spoken normally at the desk.
      const style = meta.style === undefined ? 'normal' : ['normal', 'whispered', 'far'].includes(meta.style) ? meta.style : null;
      // Word-pack collections group takes by wake word and form; older pages omit both.
      const word = meta.word === undefined ? null : clean(meta.word, 40).trim().toLowerCase();
      // Which input device recorded this take; absent means the page's original single microphone.
      const mic = meta.mic === undefined ? null : clean(meta.mic, 80).trim();
      const micDevice = meta.micDevice === undefined || meta.micDevice === null ? null : clean(meta.micDevice, 200).trim();
      const form = meta.form === undefined ? null : ['addressed', 'bare', 'confusable'].includes(meta.form) ? meta.form : null;
      // Underscores are allowed: the private roster uses owner_name/cohost_name ids.
      if (meta.word !== undefined && (!word || !/^[a-z0-9_-]{1,40}$/.test(word))) throw new RecordingError(400, 'word must be a short slug');
      if (!language || !label || !group || !phrase || !spoken || !style || !Number.isInteger(take) || take < 1 || take > 999) {
        throw new RecordingError(400, 'language/group/phrase/spoken/label/style/take must be complete');
      }
      const receivedAt = now();
      const id = receivedAt.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z') + '-' + randomUUID().slice(0, 8);
      const target = path.join(root, id);
      fs.mkdirSync(target, { mode: 0o700 });
      write(path.join(target, 'audio.wav'), wav);
      write(path.join(target, 'meta.json'), JSON.stringify({
        id, receivedAt, sessionId: meta.sessionId,
        inputKind: 'microphone', language, group, label, style, take, word, form, mic, micDevice,
        phrase: clean(meta.phrase, 200), spoken, words: clean(meta.words, 60),
        audio: { file: 'audio.wav', sha256: createHash('sha256').update(wav).digest('hex'), bytes: wav.byteLength,
                 samples, sampleRate: 16000, channels: 1, durationSeconds, peakPcm },
        capture: 'browser getUserMedia, echoCancellation/noiseSuppression/autoGainControl requested off, area-average resampled to 16 kHz',
        userAgent: clean(context.userAgent, 400),
        role: 'owner-wake-candidate', humanReview: 'pending', trainingEligible: false,
        note: 'Owner wake-phrase recording. Offering recordings is not TTS cloning, speaker enrollment, redistribution or upload consent. Review pronunciation/boundaries by listening before any detector training; reserve a whole later session as untouched final test.',
      }, null, 2) + '\n');
      return { id, durationSeconds, sha256: createHash('sha256').update(wav).digest('hex'), count: current.count + 1 };
    },
    /// Aggregate counts over the whole store. The listing is intentionally capped for
    /// the UI, so coverage must never be derived from it.
    summary() {
      ensureRoot();
      const byWord = {}, byLanguage = {}, byLabel = {}, byForm = {}, byMic = {};
      // byPhrase drives the resumable recording queue: word|language|label|style|phrase -> saved takes.
      const byPhrase = {};
      let total = 0, newest = null;
      for (const entry of fs.readdirSync(root)) {
        if (!ID.test(entry)) continue;
        let meta;
        try { meta = JSON.parse(fs.readFileSync(path.join(root, entry, 'meta.json'), 'utf8')); } catch { continue; }
        total++;
        if (!newest || meta.receivedAt > newest) newest = meta.receivedAt;
        byLabel[meta.label] = (byLabel[meta.label] ?? 0) + 1;
        if (meta.form) byForm[meta.form] = (byForm[meta.form] ?? 0) + 1;
        if (meta.word) {
          const word = (byWord[meta.word] ??= { positive: 0, negative: 0, languages: {} });
          word[meta.label === 'negative' ? 'negative' : 'positive']++;
          word.languages[meta.language] = (word.languages[meta.language] ?? 0) + 1;
          if (meta.phrase) {
            const key = `${meta.word}|${meta.language}|${meta.label}|${meta.style ?? 'normal'}|${meta.phrase}`;
            byPhrase[key] = (byPhrase[key] ?? 0) + 1;
          }
        }
        byLanguage[meta.language] = (byLanguage[meta.language] ?? 0) + 1;
        if (meta.mic) byMic[meta.mic] = (byMic[meta.mic] ?? 0) + 1;
      }
      // Usage and limits, so the page can warn before a take is refused.
      const used = usage();
      return { total, newest, byWord, byLanguage, byLabel, byForm, byPhrase, byMic,
               capacity: { count: used.count, bytes: used.bytes, maxCount, maxBytes } };
    },
    list() {
      ensureRoot();
      const rows = [];
      for (const entry of fs.readdirSync(root)) {
        if (!ID.test(entry)) continue;
        const meta = JSON.parse(fs.readFileSync(path.join(root, entry, 'meta.json'), 'utf8'));
        rows.push({ id: meta.id, receivedAt: meta.receivedAt, language: meta.language, group: meta.group,
                    label: meta.label, take: meta.take, word: meta.word ?? null, form: meta.form ?? null,
                    phrase: meta.phrase, spoken: meta.spoken, mic: meta.mic ?? null,
                    seconds: meta.audio.durationSeconds, peakPcm: meta.audio.peakPcm });
      }
      return rows.sort((a, b) => a.receivedAt < b.receivedAt ? 1 : -1).slice(0, 30);
    },
    /// Verified path to one take's WAV, or null. Never accepts a caller-supplied path.
    audioPath(id) {
      if (!ID.test(String(id))) return null;
      const file = path.join(root, String(id), 'audio.wav');
      return fs.existsSync(file) ? file : null;
    },
    /// Takes for one word (either session), newest first. Bounded for the review list.
    forWord(word, limit = 60) {
      const all = word === null;
      const slug = String(word ?? '').toLowerCase();
      if (!all && !/^[a-z0-9-]{1,40}$/.test(slug)) return [];
      ensureRoot();
      const rows = [];
      for (const entry of fs.readdirSync(root)) {
        if (!ID.test(entry)) continue;
        let meta;
        try { meta = JSON.parse(fs.readFileSync(path.join(root, entry, 'meta.json'), 'utf8')); } catch { continue; }
        if (!all && meta.word !== slug) continue;
        rows.push({ id: meta.id, receivedAt: meta.receivedAt, language: meta.language, label: meta.label,
                    word: meta.word ?? null, form: meta.form ?? null, phrase: meta.phrase, style: meta.style,
                    mic: meta.mic ?? null, seconds: meta.audio.durationSeconds, peakPcm: meta.audio.peakPcm });
      }
      return rows.sort((a, b) => (a.receivedAt < b.receivedAt ? 1 : -1)).slice(0, limit);
    },
    remove(id) {
      fs.rmSync(dir(id), { recursive: true });
      return { id, deleted: true };
    },
    /// Delete every look-alike take of one phrase (all styles). Exact field match only; the
    /// caller decides when this is allowed (the editor offers it for removed/unlisted phrases).
    removeLookalikeTakes({ word, language, phrase }) {
      ensureRoot();
      const ids = [];
      for (const entry of fs.readdirSync(root)) {
        if (!ID.test(entry)) continue;
        let meta;
        try { meta = JSON.parse(fs.readFileSync(path.join(root, entry, 'meta.json'), 'utf8')); } catch { continue; }
        if (meta.label === 'negative' && meta.word === word && meta.language === language && meta.phrase === phrase) ids.push(entry);
      }
      for (const id of ids) fs.rmSync(dir(id), { recursive: true });
      return { deleted: ids.length, ids };
    },
  };
}
