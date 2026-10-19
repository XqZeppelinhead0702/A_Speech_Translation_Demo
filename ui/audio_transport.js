// Injected inside the private UI function; no dependency or model imports.
class AudioUploadQueue {
  constructor({rate, mode, httpSend, socketSend, onPressure, onError, onAck, onRtt, now, maxPendingSeconds = 8}) {
    Object.assign(this, {rate, mode, httpSend, socketSend, onPressure, onError, onAck, onRtt, now});
    this.maxPendingSeconds = maxPendingSeconds;
    this.queue = []; this.inflight = new Map(); this.waiters = [];
    this.sequence = 0; this.acked = 0; this.pending = 0; this.maxPending = 0;
    this.working = false; this.closed = false; this.pressured = false;
  }
  add(audio) {
    if (this.closed || !audio.length) return;
    this.queue.push(audio); this.pending += audio.length / this.rate;
    this.maxPending = Math.max(this.maxPending, this.pending);
    this.pressure();
    if (this.pending > this.maxPendingSeconds) return this.fail(new Error(`未确认音频超过 ${this.maxPendingSeconds.toFixed(1)} 秒；网络未能及时恢复，已停止以避免丢失音频`));
    this.pump();
  }
  pressure() {
    const high = this.pending >= 4;
    const low = this.pending <= 1;
    if (high && !this.pressured) { this.pressured = true; this.onPressure(true); }
    if (low && this.pressured) { this.pressured = false; this.onPressure(false); }
  }
  packet() {
    // HTTP fallback merges only samples already waiting behind a slow request.
    // No extra batching delay, no audio discarded, no ASR state reset.
    const parts = []; let length = 0;
    while (this.queue.length && length + this.queue[0].length <= this.rate * 2) {
      const part = this.queue.shift(); parts.push(part); length += part.length;
    }
    if (!parts.length) throw new Error('音频块过大');
    const bytes = new Uint8Array(8 + length * 2), view = new DataView(bytes.buffer);
    const sequence = ++this.sequence;
    view.setUint32(0, sequence, true); view.setUint32(4, this.rate, true);
    let offset = 8;
    for (const part of parts) for (const value of part) {
      view.setInt16(offset, Math.round(Math.max(-1, Math.min(1, value)) * 32767), true); offset += 2;
    }
    this.inflight.set(sequence, {seconds: length / this.rate, sent: this.now()});
    return {sequence, bytes};
  }
  pump() {
    if (this.closed || this.working || !this.queue.length) return;
    if (this.mode === 'websocket') {
      try { while (this.queue.length) this.socketSend(this.packet().bytes.buffer); }
      catch (error) { this.fail(error); }
      return;
    }
    this.working = true;
    let packet;
    try { packet = this.packet(); } catch (error) { this.working = false; this.fail(error); return; }
    this.httpSend(packet).then(ack => {
      if (ack.acked_seq !== packet.sequence) throw new Error('音频块确认序号不匹配');
      this.ack(ack.acked_seq);
    }).catch(error => this.fail(error)).finally(() => {
      this.working = false; this.pump(); this.resolveWaiters();
    });
  }
  ack(sequence) {
    if (this.closed || sequence <= this.acked) return;
    if (!Number.isInteger(sequence) || sequence > this.sequence) return this.fail(new Error('音频确认序号无效'));
    const entry = this.inflight.get(sequence);
    if (entry) this.onRtt(this.now() - entry.sent);
    for (const [seq, item] of this.inflight) if (seq <= sequence) {
      this.pending = Math.max(0, this.pending - item.seconds); this.inflight.delete(seq);
    }
    this.acked = sequence; this.onAck(sequence); this.pressure(); this.resolveWaiters();
  }
  wait() {
    if (this.closed) return Promise.reject(this.error || new Error('音频传输已关闭'));
    if (!this.queue.length && !this.inflight.size && !this.working) return Promise.resolve();
    return new Promise((resolve, reject) => this.waiters.push({resolve, reject}));
  }
  resolveWaiters() {
    if (!this.queue.length && !this.inflight.size && !this.working) {
      for (const waiter of this.waiters.splice(0)) waiter.resolve();
    }
  }
  close(error = new Error('音频传输已关闭')) {
    if (this.closed) return;
    this.closed = true; this.error = error; this.queue = []; this.inflight.clear();
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
  fail(error) { if (!this.closed) { this.close(error); this.onError(error); } }
}
