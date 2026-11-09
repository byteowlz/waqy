// One shared compatible frontend, separate classifier sessions. No microphone, upload or actions.
import * as ort from '/ort/ort.wasm.min.mjs';
import { WakeEngine } from '/waqy-engine.mjs';
ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = 1;
let engine = null, busy = Promise.resolve(), epoch = 0, processedSamples = 0, modelHashes = {};

async function bytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Model asset unavailable (${response.status}): ${url}`);
  return new Uint8Array(await response.arrayBuffer());
}
async function load(releases) {
  const classifiers = Object.create(null); modelHashes = {};
  let mel, embedding, signature;
  for (const release of releases) {
    const base = `/packs/${encodeURIComponent(release)}/`;
    const response = await fetch(base + 'manifest.json');
    if (!response.ok) throw new Error(`Release unavailable: ${release}`);
    const manifest = await response.json();
    const current = `${manifest.files.mel.sha256}:${manifest.files.embedding.sha256}`;
    if (signature && signature !== current) throw new Error('These releases use different frontends. Compare them separately.');
    if (!signature) {
      signature = current;
      mel = await bytes(base + manifest.files.mel.file);
      embedding = await bytes(base + manifest.files.embedding.file);
    }
    for (const head of manifest.classifiers) {
      const key = `${release}::${head}`;
      classifiers[key] = await bytes(base + manifest.files[head].file);
      modelHashes[key] = manifest.files[head].sha256;
    }
  }
  if (!Object.keys(classifiers).length) throw new Error('Choose at least one release.');
  engine = await WakeEngine.create(ort, { mel, embedding, classifiers }, { threshold: 1.01 });
  return Object.keys(classifiers);
}
self.onmessage = ({ data }) => {
  busy = busy.then(async () => {
    try {
      if (data.type === 'load') {
        epoch = data.epoch; processedSamples = 0;
        self.postMessage({ type: 'ready', heads: await load(data.releases), modelHashes });
      } else if (data.type === 'reset' && engine) {
        epoch = data.epoch; processedSamples = 0; engine.reset();
      } else if (data.type === 'drain' && data.epoch === epoch) {
        self.postMessage({ type: 'drained', epoch });
      } else if (data.type === 'pcm' && engine && data.epoch === epoch) {
        const started = performance.now();
        const results = await engine.push(data.pcm);
        const ms = performance.now() - started;
        processedSamples += data.pcm.length;
        self.postMessage({ type: 'frame', results, ms, samples: data.pcm.length, processedSamples, epoch });
      }
    } catch (error) {
      self.postMessage({ type: 'error', message: String(error?.message ?? error) });
    }
  });
};
