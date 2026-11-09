// Replay is disabled during microphone capture so speaker playback cannot contaminate the test.
const $ = id => document.getElementById(id);
export class HeadReview {
  constructor(active, error) {
    this.active = active; this.error = error; this.recording = null; this.url = null;
    $('replay-crossing').onclick = () => this.play().catch(error);
    $('mark-crossing').onclick = () => this.recording?.mark(this.chosen()).then(() => { $('mark-result').textContent = 'Marked false trigger on this head at this sample position.'; this.render(); }).catch(error);
    $('retry-recording').onclick = () => this.recording?.retry().then(() => this.render()).catch(error);
    $('download-recording').onclick = () => this.download().catch(error);
  }
  chosen() { return this.visible?.[Number($('recorded-crossings').value)]; }
  render(recording = this.recording) {
    if (this.recording !== recording) { this.recording = recording; this.fingerprint = ''; $('recording-player').pause(); }
    const active = this.active();
    $('record-audio').disabled = active; $('recording-intent').disabled = active;
    $('recording-player').hidden = active || !this.url;
    if (active) $('recording-player').pause();
    $('recording-session').hidden = !recording;
    if (!recording) return;
    const seconds = (recording.samples / 16000).toFixed(1), saved = (recording.saved / 16000).toFixed(1);
    $('recording-status').textContent = `${active ? 'Recording' : 'Recorded'} ${seconds} s · ${saved} s saved privately · ${recording.intent}. Session ${recording.sessionId}`;
    $('recording-problem').hidden = !recording.error;
    $('recording-problem').textContent = recording.error ?? '';
    $('retry-recording').hidden = !recording.failed.length; $('retry-recording').disabled = active;
    const fingerprint = `${recording.events[0]?.atSample}:${recording.events.length}:${recording.events.at(-1)?.key}:${recording.events.at(-1)?.atSample}`;
    if (fingerprint !== this.fingerprint) {
      this.fingerprint = fingerprint; this.visible = [...recording.events].reverse();
      $('recorded-crossings').replaceChildren(...this.visible.map((event, i) => {
        const option = document.createElement('option'); option.value = i;
        option.textContent = `${(event.atSample / 16000).toFixed(2)} s · ${event.key} · ${event.score.toFixed(3)}`;
        return option;
      }));
    }
    $('replay-crossing').disabled = active || !recording.events.length;
    $('mark-crossing').disabled = !recording.events.length;
    $('download-recording').disabled = active || !recording.saved;
    $('recorded-crossings').onchange = () => { $('mark-result').textContent = ''; };
  }
  async post(route, body) {
    const response = await fetch(`/api/live/${route}`, { method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sessionId: this.recording.sessionId, ...body }) });
    if (!response.ok) throw new Error(await response.text());
    return response;
  }
  async play() {
    const event = this.chosen();
    if (this.active() || !event) return;
    const startSample = Math.max(0, event.atSample - 16000 * 2);
    const samples = Math.min(this.recording.samples, event.atSample + 16000) - startSample;
    const response = await this.post('clip', { startSample, samples });
    if (this.active()) return;
    URL.revokeObjectURL(this.url); this.url = URL.createObjectURL(await response.blob());
    const player = $('recording-player'); player.src = this.url; player.hidden = false;
    $('clip-position').textContent = `Playing ${(startSample / 16000).toFixed(2)}–${((startSample + samples) / 16000).toFixed(2)} s of the recording`;
    await player.play();
  }
  async download() {
    const response = await this.post('report', {}), index = await response.json();
    const url = URL.createObjectURL(new Blob([JSON.stringify(index, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `waqy-${this.recording.sessionId}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
