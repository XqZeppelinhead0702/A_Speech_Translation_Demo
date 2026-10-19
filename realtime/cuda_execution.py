"""Separate ASR/MT GPU queues without initializing CUDA on import or CPU tests."""
from contextlib import nullcontext
import logging
import os

LOGGER = logging.getLogger(__name__)


def make_inference_stream(device, *, priority):
    configured = os.getenv("REALTIME_CUDA_STREAMS", "1")
    if configured not in {"0", "1"}:
        raise ValueError("REALTIME_CUDA_STREAMS 必须为 0 或 1")
    if configured == "0":
        LOGGER.info("Realtime CUDA streams disabled; using compatibility execution queue.")
        return None
    import torch
    stream = torch.cuda.Stream(device=device, priority=priority)
    # Weights were loaded on the loader's stream. Establish their dependency once,
    # rather than serializing ASR and MT again on every inference/token.
    stream.wait_stream(torch.cuda.current_stream(device))
    LOGGER.info("Dedicated CUDA inference queue: device=%s requested_priority=%s", device, priority)
    return stream


def inference_stream_context(stream):
    if stream is None:
        return nullcontext()
    import torch
    return torch.cuda.stream(stream)
