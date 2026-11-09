import { PcmResampler } from '/resampler.js';

// One explicit microphone owner. Buffer only one 80 ms frame; stop fences late permission grants.
export class HeadMicrophone {
  active = false;
  pending = new Float32Array(1280);
  used = 0;
  async start(onFrame) {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Open the HTTPS link to use the microphone.');
    this.active = true;
    this.context = new AudioContext();
    try {
      await this.context.resume();
      if (!this.active) return false;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
      if (!this.active) { stream.getTracks().forEach(t => t.stop()); return false; }
      this.stream = stream;
      await this.context.audioWorklet.addModule('/capture-worklet.js');
      if (!this.active) return false;
      const resampler = new PcmResampler(this.context.sampleRate, 16000);
      this.source = this.context.createMediaStreamSource(stream);
      this.node = new AudioWorkletNode(this.context, 'moonshine-capture');
      this.mute = this.context.createGain(); this.mute.gain.value = 0;
      this.node.port.onmessage = ({ data }) => {
        if (!this.active) return;
        const pcm = resampler.feed(data);
        for (const sample of pcm) {
          this.pending[this.used++] = sample;
          if (this.used === 1280) {
            onFrame(this.pending.slice()); this.used = 0;
            if (!this.active) return;
          }
        }
      };
      this.source.connect(this.node).connect(this.mute).connect(this.context.destination);
      return true;
    } catch (error) { this.stop(); throw error; }
  }
  clearPending() { this.used = 0; }
  stop() {
    this.active = false; this.used = 0;
    this.stream?.getTracks().forEach(track => track.stop());
    this.node?.disconnect(); this.source?.disconnect();
    this.context?.close().catch(() => {});
  }
}
