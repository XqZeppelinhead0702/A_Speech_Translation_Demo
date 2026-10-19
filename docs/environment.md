# 环境、模型和示例数据

[返回 README](../README.md) · [运行指引](formal_demo.md) · [vLLM 兼容性](vllm.md)

## 环境基线与依赖选择

运行环境为 Linux、Python 3.10、NVIDIA CUDA GPU 和 FFmpeg。启动与进程清理依赖 Linux 信号，不提供 Windows 原生部署步骤。浏览器使用支持 AudioWorklet 的现代版本；远程录音需要 HTTPS 和麦克风授权。

以下为当前已使用的环境组合，不代表各模型的最低版本要求。已在现有环境运行成功时，直接沿用，不要重复安装或升级。

| 组件 | 当前基线 | 用途 |
| --- | --- | --- |
| Python | 3.10 | 其他 Python 版本需另行验证 |
| Torch / Torchaudio | 2.8.0 / 2.8.0，CUDA 12.8 wheel | 推理与重采样 |
| Transformers | 5.18.0 | Nemotron、Seamless、MMS、Hy-MT2 |
| Gradio | 6.29.1 | 两页界面与公网分享 |
| vLLM | 0.10.2 + 项目 tokenizer 兼容层 | 默认 Hy-MT2 引擎 |
| Librosa / SoXR | 0.11.0 / 1.1.0 | ASR 前处理与连续重采样 |
| FastAPI / Uvicorn / wsproto | 0.142.2 / 0.54.0 / 1.3.2 | HTTP / WebSocket |
| HTTPX | 0.28.1 | 网络检查及翻译 API |
| SpeechBrain / HyperPyYAML | 1.0.3 / 1.2.3 | 可选 CPU 语言检测 |

其余固定版本见依赖清单。Nemotron 使用较新的实验性 Transformers 接口，不应为迁就旧版 vLLM 而随意降级；项目提供局部 tokenizer 兼容层，见 [vLLM](vllm.md)。不需要安装 NeMo、fairseq2、Seamless-Streaming，也不需要 Node.js 或构建前端。

| 运行方式 | 安装清单 | 启动选择 |
| --- | --- | --- |
| 默认两页 + Hy-MT2/vLLM | `requirements-vllm.txt`，包含基础、实时及引擎依赖 | 默认参数 |
| 两页 + 原生 Hy-MT2 / Seamless 文本翻译 | `requirements.txt` + `requirements-streaming.txt`，另装 Torch/Torchaudio | `--mt-backend hy-mt2` 或 `seamless` |
| 仅文件端到端翻译 | `requirements.txt`，另装 Torch/Torchaudio | `--no-realtime --file-translation speech` |
| 无 GPU 界面预览 | 基础依赖；预览实时页还需实时依赖，无需 vLLM 引擎 | `--demo-mode`，不加载模型 |

### 从零安装示例

先核对 GPU 节点驱动和 CUDA wheel 兼容性。以下只用于新环境，启动脚本不执行安装；已有环境无需重复运行：

```bash
conda create -n st_demo python=3.10 -y
conda activate st_demo
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-vllm.txt
conda install -c conda-forge ffmpeg -y
python -m pip check
ffmpeg -version
```

Torch 与 Torchaudio 必须匹配；vLLM 预编译 wheel 也与 Torch/CUDA 构建有关。服务器无法满足此组合时，先参照 [vLLM 版本说明](vllm.md)核对，或使用原生 Hy-MT2；不要只替换一个二进制包。

不用 vLLM 时，将完整清单安装命令改为 `python -m pip install -r requirements.txt -r requirements-streaming.txt`。`python app.py --help` 可在 CPU 上查看参数；集群的真实模型运行全部通过 Slurm 提交。

## 四个模型的本地部署

默认两页需要以下完整目录；权重不随应用分发，运行时不会自动下载：

| 用途 | 默认目录 | 官方仓库 |
| --- | --- | --- |
| 文件转写 / 端到端翻译 | `ckpts/seamless-m4t-v2-large` | [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) |
| 文件语言检测 | `ckpts/mms-lid-256` | [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) |
| 流式 / 文件级联识别 | `ckpts/nemotron-3.5-asr-streaming-0.6b` | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| 共用文本翻译 | `ckpts/Hy-MT2-1.8B` | [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)，可换 [7B](https://huggingface.co/tencent/Hy-MT2-7B) |

在能联网的位置下载，保留权重分片、配置、generation config、processor/tokenizer 等全部配套文件：

```bash
python - <<'DOWNLOAD_MODEL'
from huggingface_hub import snapshot_download
for repo, directory in [
    ("facebook/seamless-m4t-v2-large", "seamless-m4t-v2-large"),
    ("facebook/mms-lid-256", "mms-lid-256"),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b"),
    ("tencent/Hy-MT2-1.8B", "Hy-MT2-1.8B"),
]:
    snapshot_download(repo_id=repo, local_dir=f"ckpts/{directory}")
DOWNLOAD_MODEL
```

下载不会运行推理。计算节点不能联网时，将完整目录复制到共享存储，在 `.env.local` 指定路径：

```bash
DEMO_MODEL_DIR=/path/to/seamless-m4t-v2-large
DEMO_LID_MODEL_DIR=/path/to/mms-lid-256
REALTIME_ASR_MODEL_DIR=/path/to/nemotron-3.5-asr-streaming-0.6b
REALTIME_HY_MT_MODEL_DIR=/path/to/Hy-MT2-1.8B
# 另外配置 REALTIME_VLLM_MODEL_DIR 时，它会优先用于 vLLM。
```

Seamless 默认 float16，MMS 和 Nemotron 默认 float32，Hy-MT2 默认 bfloat16。这些精度分别配置；BF16 需 GPU 支持。1.8B / 7B 共用适配器，但服务与客户端必须匹配模型及 tokenizer，见 [Hy-MT2](hy_mt_testing.md)。

默认 Slurm 申请 **1 GPU、8 CPU、64 GB 主机内存**。建议先使用 A100 80 GB 验证整体部署。vLLM 默认按整卡显存的 40% 制定预算，其他模型另需空间；这不是整个应用的显存上限或峰值实测。

### 精简运行

- 仅文件端到端翻译：`--no-realtime --file-translation speech`，只需 SeamlessM4T 和所选语言检测模型。
- 只关闭实时页但文件页选择级联：`--no-realtime --file-translation text`，仍需 Nemotron 和文本翻译模型。
- CPU 语言检测：设置 `DEMO_LID_BACKEND=speechbrain`，准备 `ckpts/lang-id-voxlingua107-ecapa/`，清除或改正固定的 MMS 路径。不会同时加载 MMS，见[语言检测](language_detection.md)。

## 示例音频

两页共用 `data/test_samples/realtime_samples.json`，展示 10 条约 15–27 秒的真人朗读，覆盖 8 种输入语言。MP3 随项目提供，无需另装 `datasets` 或重新下载。

文件页填入样本后仍需提交；实时页按真实播放时钟上传。新样本没有标准参考译文，CPU 演示模式只提供音频预览或明确标注的 MOCK 字幕，不能评估模型质量或时延。

旧 CoVoST 短样本和 90 秒长样本不在默认清单中；保留的文件仍可手动使用，但文件页默认上限 60 秒。来源和裁剪信息见[音频来源](../data/test_samples/SOURCES.md)，验证方法见[实时页面验证](realtime_testing.md)。`datasets` 仅为重新下载旧 CoVoST 数据时的可选工具依赖。
