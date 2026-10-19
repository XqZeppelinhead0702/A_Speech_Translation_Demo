"""PCM validation and normalization, without model imports."""

import numpy as np


def normalize_audio(audio, sample_rate):
    if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("采样率必须为 8000–192000 Hz。")
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2) or (audio.ndim == 2 and not 1 <= audio.shape[1] <= 8):
        raise ValueError("音频需要单声道或最多 8 声道的数组。")
    if len(audio) > sample_rate * 5:
        raise ValueError("每个上传块最多 5 秒；请使用较短发送间隔。")
    if audio.dtype.kind in "iu":
        info = np.iinfo(audio.dtype)
        scale = (info.max - info.min + 1) / 2
        audio = (audio.astype(np.float32) - (scale if info.min == 0 else 0)) / scale
    elif audio.dtype.kind == "f":
        audio = audio.astype(np.float32, copy=False)
    else:
        raise ValueError("不支持该音频数据类型。")
    if not np.isfinite(audio).all() or np.any(np.abs(audio) > 1.01):
        raise ValueError("音频含无效采样或未归一化的浮点幅值。")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)

