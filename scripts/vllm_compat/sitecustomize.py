"""Process-local compatibility for vLLM 0.10.2 with Transformers 5.

Only the engine launcher prepends this directory to PYTHONPATH. Python spawn
workers inherit it, so the tokenizer's pickle/reconstruction path is covered too.
No site-packages files are changed; no vLLM import, model load or CUDA call here.
"""
import importlib.metadata
import os


def apply_tokenizer_compat():
    try:
        vllm_version = importlib.metadata.version("vllm")
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return False
    if vllm_version.split("+")[0] != "0.10.2" or transformers_version.split(".")[0] != "5":
        return False
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return False
    # vLLM caches and exposes this list; it does not need AddedToken objects here.
    # TF5's all_special_tokens remains the public list of special token strings.
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda tokenizer: list(tokenizer.all_special_tokens))
    return True


if os.getenv("SPEECH_DEMO_VLLM_COMPAT", "0") == "1":
    apply_tokenizer_compat()
