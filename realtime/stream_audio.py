"""Continuous resampling and exact STFT windows; never split semantic sentences."""
import numpy as np


class StreamCancelled(Exception):
    pass


def resample_stream(chunks, output_rate=16000):
    """Preserve SoXR filter state and flush its delayed samples once at EOF."""
    converter, rate = None, None
    for audio, sample_rate in chunks:
        if rate is None:
            rate = sample_rate
            if rate != output_rate:
                import soxr
                converter = soxr.ResampleStream(rate, output_rate, 1, dtype="float32", quality="HQ")
        if sample_rate != rate:
            raise ValueError("会话中采样率不能变化")
        value = converter.resample_chunk(audio) if converter else audio
        if len(value):
            yield np.ascontiguousarray(value, dtype=np.float32)
    if converter:
        value = converter.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        if len(value):
            yield value


class PCMWindows:
    """Windows align to the full signal's centered STFT, including the short tail.

    windows yields (pcm, first, real_frames, observed_pcm, end_sample, last). FFT
    overlap supplies acoustic context only; decoder state survives every window.
    """
    def __init__(self, chunks, *, first_frames, frames, hop=160, n_fft=512):
        self.chunks = iter(chunks)
        self.first_frames, self.frames, self.hop, self.n_fft = first_frames, frames, hop, n_fft
        self.buffer = np.empty(0, dtype=np.float32)
        self.base = self.total = self.frame_index = self.observed = 0
        self.eof = False

    def __iter__(self):
        first = True
        while True:
            count = self.first_frames if first else self.frames
            start = 0 if first else self.frame_index * self.hop - self.n_fft // 2
            end = (self.frame_index + count - 1) * self.hop + self.n_fft // 2 + 1
            while not self.eof and self.total < end:
                try:
                    block = next(self.chunks)
                except StopIteration:
                    self.eof = True
                else:
                    self.buffer = np.concatenate((self.buffer, block))
                    self.total += len(block)
            if not self.total:
                return
            if not first and self.eof and self.frame_index >= self.total // self.hop:
                if self.total > self.observed:
                    tail = self.buffer[self.observed - self.base:].copy()
                    yield np.empty(0, dtype=np.float32), False, 0, tail, self.total, True
                return
            last = self.eof and end > self.total
            real_frames = max(0, min(count, self.total // self.hop - self.frame_index)) if last else count
            stop = min(end, self.total)
            pcm = self.buffer[max(0, start) - self.base:stop - self.base].copy()
            if start < 0:
                pcm = np.pad(pcm, (-start, 0))
            observed_end = self.total if last else min(self.total, (self.frame_index + count) * self.hop)
            observed = self.buffer[self.observed - self.base:observed_end - self.base].copy()
            yield pcm, first, real_frames, observed, observed_end, last
            self.observed = observed_end
            self.frame_index += count
            first = False
            keep = max(0, self.frame_index * self.hop - self.n_fft // 2)
            # Retain the overlap for the next STFT, not an ever-growing waveform.
            discard = min(keep, self.total) - self.base
            self.buffer = self.buffer[discard:].copy()
            self.base += discard


class SilenceMeter:
    def __init__(self, rate=16000, threshold=0.008):
        self.rate, self.threshold = rate, threshold
        self.pending = np.empty(0, dtype=np.float32)
        self.silent = 0

    def feed(self, audio):
        self.pending = np.concatenate((self.pending, audio))
        size = round(self.rate * .02)
        while len(self.pending) >= size:
            frame, self.pending = self.pending[:size], self.pending[size:]
            self.silent = self.silent + size if np.sqrt(np.mean(frame * frame)) < self.threshold else 0
        self.pending = self.pending.copy()
        return self.silent / self.rate
