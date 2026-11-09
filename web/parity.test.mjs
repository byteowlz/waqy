// Streaming WakeEngine (onnxruntime-web, wasm backend) vs the Python batch reference.
// Reference file: PARITY_REF=<json from training/parity_ref.py>, models in WAKE_MODELS dir.
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { FRAME, WakeEngine } from './waqy-engine.mjs';

const refPath = process.env.PARITY_REF;
const modelDir = process.env.WAKE_MODELS ?? new URL('../training/frontend/', import.meta.url).pathname;
ort.env.wasm.numThreads = 1;

test.skipIf(!refPath)('streaming scores track the batch reference', async () => {
  const ref = JSON.parse(readFileSync(refPath, 'utf8'));
  const engine = await WakeEngine.create(ort, {
    mel: readFileSync(`${modelDir}/melspectrogram_raw.onnx`),
    embedding: readFileSync(`${modelDir}/embedding_model.onnx`),
    classifiers: { wake: readFileSync(ref.classifier) },
  }, { dither: 0 }); // the Python reference has no dither
  for (const c of ref.cases) {
    engine.reset();
    const audio = Float32Array.from(c.audio);
    const stream = [];
    for (let i = 0; i < audio.length; i += FRAME) {
      for (const r of await engine.push(audio.subarray(i, i + FRAME))) stream.push(r.scores.wake);
    }
    const batchMax = Math.max(...c.scores);
    const streamMax = Math.max(...stream);
    // Embedding windows sit on the same 80 ms grid; allow the engine's warm-up offset.
    let best = Infinity;
    for (let shift = -3; shift <= 3; shift++) {
      let err = 0, n = 0;
      for (let i = 0; i < stream.length; i++) {
        const j = i + (c.scores.length - stream.length) + shift;
        if (j >= 0 && j < c.scores.length) { err = Math.max(err, Math.abs(stream[i] - c.scores[j])); n++; }
      }
      if (n >= Math.min(5, stream.length)) best = Math.min(best, err);
    }
    console.log(`${c.wav}: batch max ${batchMax.toFixed(4)} stream max ${streamMax.toFixed(4)} aligned max |diff| ${best.toFixed(4)} (${stream.length} vs ${c.scores.length} frames)`);
    expect(Math.abs(batchMax - streamMax)).toBeLessThan(0.05);
    expect(best).toBeLessThan(0.05);
  }
}, 120000);
