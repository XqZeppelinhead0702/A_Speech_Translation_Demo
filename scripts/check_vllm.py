"""CPU-only vLLM/config diagnostics. Does not import vLLM or load model weights."""
import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("ckpts/Hy-MT2-1.8B"))
    parser.add_argument("--gpu-info", action="store_true", help="只查询 nvidia-smi 驱动/显卡信息，不运行推理")
    args = parser.parse_args()
    report = dict(python=sys.version.split()[0], executable=sys.executable,
                  platform=platform.system(), machine=platform.machine(), libc=platform.libc_ver(),
                  packages={}, errors=[], warnings=[])
    for package in ["vllm", "torch", "transformers", "tokenizers", "httpx"]:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    if not report["packages"]["vllm"]:
        report["errors"].append("此 Python 未安装 vLLM；默认使用 DEMO_PYTHON，可选 REALTIME_VLLM_PYTHON 指定已有环境")
    else:
        # Read, rather than import, the registry. Importing vLLM can touch CUDA.
        dist = metadata.distribution("vllm")
        registry = Path(dist.locate_file("vllm/model_executor/models/registry.py"))
        report["architecture_listed"] = registry.is_file() and "HunYuanDenseV1ForCausalLM" in registry.read_text()
        if not report["architecture_listed"]:
            report["warnings"].append("注册表未直接列出 HunYuanDenseV1ForCausalLM；可能需模型后端支持或更新，不能凭版本号确认")
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda  # no cuda.is_available()/initialization
    except Exception as exc:
        report["errors"].append(f"Torch CPU 导入失败：{type(exc).__name__}: {exc}")
    try:
        config = json.loads((args.model_dir / "config.json").read_text())
        report["model"] = {key: config.get(key) for key in ["model_type", "architectures", "torch_dtype", "max_position_embeddings"]}
        if config.get("model_type") != "hunyuan_v1_dense" or config.get("quantization_config"):
            raise ValueError("仅适配原始 Hy-MT2-1.8B/7B dense 权重")
        from transformers import AutoConfig, AutoTokenizer
        AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("缺少聊天模板")
        report["tokenizer_probe_ids"] = tokenizer.encode("Hello，测试。", add_special_tokens=False)
        report["model_config_tokenizer_ok"] = True
        report["tokenizer_legacy_special_tokens_property"] = hasattr(tokenizer, "all_special_tokens_extended")
        if (report["packages"]["vllm"] == "0.10.2"
                and str(report["packages"]["transformers"]).startswith("5.")
                and not report["tokenizer_legacy_special_tokens_property"]):
            report["warnings"].append("vLLM 0.10.2 的 tokenizer 缓存依赖 TF5 已删除的属性；serve_vllm.sh 默认启用项目内兼容层，不修改环境")
    except Exception as exc:
        report["errors"].append(f"本地配置/tokenizer 读取失败：{type(exc).__name__}: {exc}")
    if args.gpu_info:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10)
            report["gpu_query"] = result.stdout.strip() if result.returncode == 0 else "不可用（可能位于无 GPU 的登录节点）"
        except (OSError, subprocess.TimeoutExpired):
            report["gpu_query"] = "不可用；请在 Slurm GPU 作业中查询"
    report["warnings"].append("此检查不加载权重、不执行 CUDA kernel，不能证明 GPU 推理或旧版 vLLM 完整兼容；仍需 Slurm 冒烟测试")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
