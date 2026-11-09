// Wake detector worker: onnxruntime-web (WASM, CPU, single thread) + the streaming WakeEngine.
// Receives 16 kHz Float32 PCM, returns per-80 ms scores. Audio never leaves this worker.
import * as ort from '/ort/ort.wasm.min.mjs';
import { WakeEngine } from '/waqy-engine.mjs';

ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = 1;

let engine = null;
let busy = Promise.resolve();

async function load({ manifest, threshold }) {
  // A pack (any published release) serves its own verified files; the installed release keeps /models/.
  const base = manifest.pack ? `/packs/${encodeURIComponent(manifest.pack)}` : '/models';
  const fetchModel = async (name) => {
    const response = await fetch(`${base}/${manifest.files[name].file}`);
    if (!response.ok) throw new Error(`${name}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const classifiers = {};
  for (const id of manifest.classifiers) classifiers[id] = await fetchModel(id);
  engine = await WakeEngine.create(ort, { mel: await fetchModel('mel'), embedding: await fetchModel('embedding'), classifiers }, { threshold });
}

self.onmessage = ({ data }) => {
  busy = busy.then(async () => {
    try {
      if (data.type === 'load') {
        await load(data);
        self.postMessage({ type: 'ready' });
      } else if (data.type === 'threshold' && engine) {
        engine.threshold = data.threshold;
      } else if (data.type === 'playback' && engine) {
        engine.setPlayback(data.active);
      } else if (data.type === 'reset' && engine) {
        engine.reset();
      } else if (data.type === 'pcm' && engine) {
        const started = performance.now();
        const results = await engine.push(data.pcm);
        if (results.length) self.postMessage({ type: 'scores', results, ms: (performance.now() - started) / results.length });
      }
    } catch (error) {
      self.postMessage({ type: 'error', message: String(error?.message ?? error) });
    }
  });
};
