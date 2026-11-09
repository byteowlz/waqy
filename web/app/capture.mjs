import { PcmResampler } from '/resampler.js';

export const MAX_SECONDS = 60;
export async function decodeFile(file) {
  if (!file || file.size > 20 * 1024 * 1024) throw new Error('Choose an audio file smaller than 20 MB.');
  const context = new OfflineAudioContext(1, 1, 16000);
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  if (!decoded.length || decoded.duration > MAX_SECONDS) throw new Error('Choose a clip between 0 and 60 seconds.');
  const pcm = new Float32Array(decoded.length);
  for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
    const data = decoded.getChannelData(channel);
    for (let i = 0; i < pcm.length; i++) pcm[i] += data[i] / decoded.numberOfChannels;
  }
  return pcm;
}

/** One explicit microphone owner. Cancellation also fences a late permission grant. */
export class Microphone {
  epoch = 0;
  active = false;
  deviceId = null;
  async start({ onProgress = () => {}, onLimit = () => {}, maxSeconds = MAX_SECONDS, deviceId = null } = {}) {
    if (this.active) throw new Error('Microphone is already active.');
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Open the HTTPS link to record, or choose an audio file.');
    this.active = true;
    const epoch = ++this.epoch;
    this.parts = [];
    this.samples = 0;
    this.maxSeconds = maxSeconds;
    try {
      const context = new AudioContext();
      this.context = context;
      await context.resume();
      // One Microphone instance owns one input device, so several can run at once.
      const audio = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
      if (deviceId) audio.deviceId = { exact: deviceId };
      const stream = await navigator.mediaDevices.getUserMedia({ audio, video: false });
      this.deviceId = deviceId;
      if (epoch !== this.epoch) { stream.getTracks().forEach(track => track.stop()); return false; }
      this.stream = stream;
      await context.audioWorklet.addModule('/capture-worklet.js');
      if (epoch !== this.epoch) return false;
      this.resampler = new PcmResampler(context.sampleRate, 16000);
      this.source = context.createMediaStreamSource(stream);
      this.node = new AudioWorkletNode(context, 'moonshine-capture');
      this.mute = context.createGain();
      this.mute.gain.value = 0;
      this.node.port.onmessage = ({ data }) => {
        if (epoch !== this.epoch || !this.active) return;
        const room = this.maxSeconds * 16000 - this.samples;
        const pcm = this.resampler.feed(data).subarray(0, room);
        this.parts.push(pcm);
        this.samples += pcm.length;
        onProgress(this.samples / 16000);
        if (this.samples >= this.maxSeconds * 16000) onLimit();
      };
      this.source.connect(this.node).connect(this.mute).connect(context.destination);
      this.timer = setTimeout(onLimit, this.maxSeconds * 1000);
      return true;
    } catch (error) {
      if (epoch === this.epoch) this.cancel();
      if (epoch !== this.epoch && error?.name === 'AbortError') return false;
      throw error;
    }
  }
  /** Copy the PCM recorded so far without closing the microphone (back-to-back takes). */
  interim() {
    const pcm = new Float32Array(this.samples);
    let offset = 0;
    for (const part of this.parts ?? []) { pcm.set(part, offset); offset += part.length; }
    return pcm;
  }
  /** PCM recorded since `fromSeconds` (whole frames). One take window, no cumulative send. */
  interimSlice(fromSeconds = 0) {
    const pcm = this.interim();
    const offset = Math.max(0, Math.min(pcm.length, Math.floor(fromSeconds * 16000)));
    return pcm.subarray(offset);
  }
  stop() {
    const pcm = new Float32Array(this.samples);
    let offset = 0;
    for (const part of this.parts ?? []) { pcm.set(part, offset); offset += part.length; }
    this.cancel();
    return pcm;
  }
  cancel() {
    ++this.epoch;
    this.active = false;
    clearTimeout(this.timer);
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect();
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); }
    this.mute?.disconnect();
    if (this.context && this.context.state !== 'closed') void this.context.close().catch(() => {});
    this.stream = this.source = this.node = this.mute = this.context = null;
    this.parts = [];
    this.samples = 0;
  }
}

// Input devices (labels only appear after the page has microphone permission once).
export async function listInputs() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(device => device.kind === 'audioinput')
    .map((device, index) => ({ deviceId: device.deviceId || null, label: device.label || `Microphone ${index + 1}` }));
}

/**
 * Cut a take to its speech span with a margin, so takes from devices with different start
 * latencies (a Bluetooth headset can lag 100-200 ms) align on the phrase. Returns the input
 * when nothing voiced is found, so a quiet take is still saved and flagged elsewhere.
 */
export function trimToSpeech(pcm, marginSeconds = 0.15) {
  const frame = 320, n = Math.floor(pcm.length / frame);
  if (!n) return pcm;
  const energy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = i * frame; j < (i + 1) * frame; j++) sum += pcm[j] * pcm[j];
    energy[i] = Math.sqrt(sum / frame);
  }
  let peak = 0;
  for (const value of energy) peak = Math.max(peak, value);
  if (peak <= 0) return pcm;
  const floor = Math.max(peak * 0.05, 1e-5);
  let first = 0, last = n - 1;
  while (first < n && energy[first] <= floor) first++;
  while (last > first && energy[last] <= floor) last--;
  if (first >= last) return pcm;
  const margin = Math.round(marginSeconds * 16000);
  const start = Math.max(0, first * frame - margin), end = Math.min(pcm.length, (last + 1) * frame + margin);
  return pcm.subarray(start, end);
}

export function wavBlob(pcm) {
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, pcm[i])) * 32767), true);
  return new Blob([buffer], { type: 'audio/wav' });
}
