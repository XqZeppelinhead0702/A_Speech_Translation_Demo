// Lightweight acoustic endpointing, using sample time rather than network or
// rendering timers. This only ends microphone input; it never segments subtitles.
class AudioEndpointDetector {
  constructor({rate, silenceSeconds = 2.5}) {
    this.rate = rate;
    this.limit = silenceSeconds;
    this.frameSize = Math.max(1, Math.round(rate * .02));
    this.count = 0; this.sum = 0; this.squares = 0;
    this.noise = .001; this.evidence = 0; this.armed = false;
    this.quiet = 0; this.ended = false;
  }
  feed(audio) {
    if (this.ended) return false;
    for (const value of audio) {
      this.sum += value; this.squares += value * value;
      if (++this.count < this.frameSize) continue;
      // Remove DC bias, which can otherwise look like continuous speech.
      const rms = Math.sqrt(Math.max(0, this.squares / this.count - (this.sum / this.count) ** 2));
      const seconds = this.count / this.rate;
      this.count = 0; this.sum = 0; this.squares = 0;
      const voiced = rms >= Math.max(.006, this.noise * 3);
      if (voiced) {
        this.evidence += seconds;
        this.quiet = 0;
        // A single click/pop must not arm automatic stopping.
        if (this.evidence >= .2 - 1e-9) this.armed = true;
      } else {
        this.noise += .04 * (rms - this.noise);
        if (!this.armed) this.evidence = Math.max(0, this.evidence - seconds * .5);
        else this.quiet += seconds;
      }
      if (this.armed && this.quiet >= this.limit - 1e-9) {
        this.ended = true;
        return true;
      }
    }
    return false;
  }
}
