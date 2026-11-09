// Self-trigger: our own TTS output must not wake the detector, and the gate must not break real speech.
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { FRAME, WakeEngine } from './waqy-engine.mjs';
ort.env.wasm.numThreads = 1;

const dir = process.env.WAKE_MODELS ?? `${process.env.HOME}/.local/share/foxline/wake-models/spqx-gpu5`;
const bank = process.env.WAKE_TTS_BANK ?? `${process.env.HOME}/.local/share/foxline/wake-tts-1/voice-bank-2/wav`;
const models = { mel: readFileSync(`${dir}/melspectrogram_raw.onnx`), embedding: readFileSync(`${dir}/embedding_model.onnx`),
  classifiers: { hey_oqto: readFileSync(`${dir}/hey_oqto.onnx`), oqto: readFileSync(`${dir}/oqto.onnx`) } };
const existing = (f) => { try { readFileSync(f); return f; } catch { return null; } };
const ttsSpeaksWake = existing(`${bank}/de-ctx-0002.wav`);
const humanSaysWake = existing(`${bank}/de-ctx-0005.wav`);

function wav(path) {
  const b = readFileSync(path); const body = new Float32Array((b.length - 44) / 2);
  for (let i = 0; i < body.length; i++) body[i] = b.readInt16LE(44 + i * 2) / 32768;
  // 1.5 s of quiet room before the utterance, so the engine has 16 frames of history when it starts
  const out = new Float32Array(body.length + 16000 * 2);
  out.set(body, 16000 * 1.5);
  return out;
}
async function scan(pcm, { gate = false } = {}) {
  const engine = await WakeEngine.create(ort, models);
  let peak = 0, fired = 0;
  if (gate) engine.setPlayback(true); // the app starts speaking at t=0
  for (let i = 0; i < pcm.length; i += FRAME) {
    for (const r of await engine.push(pcm.subarray(i, i + FRAME))) {
      const s = Math.max(r.scores.hey_oqto, r.scores.oqto);
      peak = Math.max(peak, s);
      if (r.detections.length) fired++;
    }
  }
  return { peak: +peak.toFixed(3), fired };
}

test.skipIf(!ttsSpeaksWake || !humanSaysWake)('own playback does not wake it, and the gate still hears a person', async () => {
  const spoken = await scan(wav(ttsSpeaksWake));
  const gated = await scan(wav(ttsSpeaksWake), { gate: true });
  const person = await scan(wav(humanSaysWake));
  console.log(`tts open ${JSON.stringify(spoken)} | tts gated ${JSON.stringify(gated)} | person open ${JSON.stringify(person)}`);
  expect(spoken.fired).toBeGreaterThan(0); // the failure this gate exists for
  expect(gated.fired).toBe(0);
  expect(gated.peak).toBe(0);
  expect(person.fired).toBeGreaterThan(0); // gating must not be a blanket deafness
}, 120000);

test.skipIf(!ttsSpeaksWake || !humanSaysWake)('the gate releases after playback, drops only the echo tail, and keeps the clock', async () => {
  const engine = await WakeEngine.create(ort, models);
  const feed = async (pcm) => { const out = []; for (let i = 0; i < pcm.length; i += FRAME) out.push(...await engine.push(pcm.subarray(i, i + FRAME))); return out; };
  engine.setPlayback(true);
  const during = await feed(wav(ttsSpeaksWake));
  const framesAfterPlayback = engine.frameIndex;
  const resumeAt = engine.setPlayback(false, { tailMs: 400 });
  expect(resumeAt).toBe(framesAfterPlayback + 5);
  const after = await feed(wav(humanSaysWake));
  expect(during.length).toBe(0);
  expect(after[0].frame).toBeGreaterThanOrEqual(resumeAt); // tail frames were dropped, the clock kept running
  expect(after.some((r) => r.detections.length)).toBe(true); // and it hears a person again
});
