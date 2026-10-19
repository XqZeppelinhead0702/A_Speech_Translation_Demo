"""Optional text translator selection, shared by service and test entrypoints."""
import os


def create_text_translator(*, backend=None, model_dir=None, precision=None,
                           max_source_tokens=None, max_new_tokens=None, sampling=None,
                           existing_translator=None, api_base=None, served_model=None,
                           require_slurm=True):
    backend = backend or os.getenv("REALTIME_MT_BACKEND", "seamless")
    if backend not in {"seamless", "hy-mt2", "hy-mt2-vllm"}:
        raise ValueError("REALTIME_MT_BACKEND 必须为 seamless、hy-mt2 或 hy-mt2-vllm")
    if backend != "hy-mt2-vllm" and (api_base is not None or served_model is not None):
        raise ValueError("--mt-api-base / --mt-served-model 仅适用于 hy-mt2-vllm")
    if backend in {"hy-mt2", "hy-mt2-vllm"}:
        if existing_translator is not None:
            raise ValueError("Hy-MT2 不可复用 SeamlessM4T 的权重；请独立加载所选文本模型。")
        value = os.getenv("REALTIME_HY_MT_SAMPLING", "0") if sampling is None else sampling
        if value not in {True, False, "0", "1"}:
            raise ValueError("REALTIME_HY_MT_SAMPLING 必须为 0 或 1")
        if backend == "hy-mt2-vllm":
            from .vllm_mt import VLLMTextTranslator
            adapter = VLLMTextTranslator
            extra = dict(api_base=api_base or os.getenv("REALTIME_VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
                         served_model=served_model or os.getenv("REALTIME_VLLM_SERVED_MODEL", "hy-mt2"))
            model_dir = model_dir or os.getenv("REALTIME_VLLM_MODEL_DIR")
        else:
            from .hy_mt import HyMTTextTranslator
            adapter, extra = HyMTTextTranslator, {"require_slurm": require_slurm}
        return adapter(model_dir or os.getenv("REALTIME_HY_MT_MODEL_DIR"),
            precision or os.getenv("REALTIME_HY_MT_PRECISION", "bfloat16"),
            max_source_tokens=int(max_source_tokens if max_source_tokens is not None else os.getenv("REALTIME_HY_MT_MAX_SOURCE_TOKENS", "4096")),
            max_new_tokens=int(max_new_tokens if max_new_tokens is not None else os.getenv("REALTIME_HY_MT_MAX_NEW_TOKENS", "1024")),
            sampling=value in {True, "1"}, **extra)
    from .backend import SeamlessTextTranslator
    if sampling:
        raise ValueError("--mt-sampling 仅适用于 Hy-MT2")
    return SeamlessTextTranslator(model_dir or os.getenv("DEMO_MODEL_DIR"),
        precision or os.getenv("DEMO_PRECISION", "float16"),
        max_source_tokens=int(max_source_tokens if max_source_tokens is not None else os.getenv("REALTIME_T2T_MAX_SOURCE_TOKENS", "1024")),
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None else os.getenv("REALTIME_T2T_MAX_NEW_TOKENS", "1024")),
        existing_translator=existing_translator)


def add_translation_arguments(parser):
    parser.add_argument("--mt-backend", choices=["seamless", "hy-mt2", "hy-mt2-vllm"], default=None,
                        help="文本翻译后端；默认读 REALTIME_MT_BACKEND，否则 seamless")
    parser.add_argument("--mt-model-dir", help="所选文本模型的完整本地目录")
    parser.add_argument("--mt-precision", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--mt-api-base", help="vLLM API，例如 http://127.0.0.1:8000/v1")
    parser.add_argument("--mt-served-model", help="vLLM --served-model-name；默认 hy-mt2")
    parser.add_argument("--mt-max-source-tokens", type=int, help="输入上限；Hy-MT2 包括提示词和聊天模板")
    parser.add_argument("--mt-max-new-tokens", type=int)
    parser.add_argument("--mt-sampling", action="store_true", default=None,
                        help="Hy-MT2 使用官方推荐采样参数；默认贪心生成，降低反复翻译的随机波动")


def translation_options(args):
    return dict(backend=args.mt_backend, model_dir=args.mt_model_dir, precision=args.mt_precision,
                max_source_tokens=args.mt_max_source_tokens, max_new_tokens=args.mt_max_new_tokens,
                sampling=args.mt_sampling, api_base=args.mt_api_base, served_model=args.mt_served_model)
