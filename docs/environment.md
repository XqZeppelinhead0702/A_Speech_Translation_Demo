# 环境、模型和示例数据

[返回 README](../README.md)

## 环境要求

部署环境为 Linux、Python 3.10+、NVIDIA CUDA GPU 和 FFmpeg。应用使用 Linux 信号控制启动超时，不提供 Windows 原生部署步骤。

当前依赖基线：

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10 |
| torch / torchaudio | 2.6.0，CUDA 12.4 wheel |
| Gradio | 6.29.1 |
| Transformers | 5.18.0 |
| SpeechBrain / HyperPyYAML | 1.0.3 / 1.2.3 |
| numpy / soundfile / sentencepiece | 以 `requirements.txt` 为准 |

先进入项目根目录，准备环境：

```bash
conda create -n st_demo python=3.10 -y
conda activate st_demo
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
ffmpeg -version
```

CUDA wheel 需要与机器驱动匹配；已有匹配的 torch/torchaudio 时不必重复安装。集群模块及 Conda 初始化路径以实际环境为准，不需要为了运行 Demo 重新编译 PyTorch。

## 准备模型权重

默认路径：

```text
ckpts/seamless-m4t-v2-large/
├── config.json
├── generation_config.json
├── model.safetensors.index.json
├── model-*.safetensors
└── processor/tokenizer 等配套文件
```

请保留官方仓库的完整配套文件。可以复制已有的完整目录，也可以在允许联网的位置执行以下下载命令；此步骤不加载模型或运行 GPU 推理：

```bash
python - <<'DOWNLOAD_MODEL'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/seamless-m4t-v2-large",
    local_dir="ckpts/seamless-m4t-v2-large",
)
DOWNLOAD_MODEL
```

下载机器与运行机器不同时，把完整目录复制到运行机器可读的位置。在 `.env.local` 设置 `DEMO_MODEL_DIR=/path/to/model` 可使用其他目录。程序以 `local_files_only=True` 加载，运行时不自动补下载缺失文件。

模型信息和使用许可见[官方模型卡](https://huggingface.co/facebook/seamless-m4t-v2-large)。

默认还需要准备本地 `ckpts/mms-lid-256/`，包含 `config.json`、`preprocessor_config.json` 和分类模型权重；通过现有 Transformers 依赖加载，常驻与翻译模型相同的 GPU。原来的 `ckpts/lang-id-voxlingua107-ecapa/` 保留，用 `--lid-backend speechbrain` 可切回 CPU 检测。配置与离线准备步骤见[自动语言检测](language_detection.md)。

## 示例音频

示例 MP3 和 `references.csv` 已在 `data/test_samples/` 中提供，使用 Demo 无需再次下载。页面使用以下三段：

| 文件 | 输入语言 | 输出语言 |
| --- | --- | --- |
| `zh-CN_en_04.mp3` | 中文普通话 | 英语 |
| `en_zh-CN_01.mp3` | 英语 | 中文普通话 |
| `fr_en_04.mp3` | 法语 | 英语 |

示例输入语言均设为“自动检测”。点击示例只填入输入，不会启动时预先推理。参考文本不作为模型结果显示。若需要重新获取数据，可选执行：

```bash
python -m pip install datasets
cd data
python download_covost2_test_samples.py
cd ..
```

下载工具从 [`fixie-ai/covost2`](https://huggingface.co/datasets/fixie-ai/covost2) 的三个语言对各读取五个测试片段。`datasets` 只用于这个工具，不是应用运行依赖。

## 模型调用方式

应用使用 `AutoProcessor` 和 `SeamlessM4Tv2Model`，默认 float16。音频先混合为单声道，再使用 torchaudio 重采样至 16 kHz。

对同一段音频：

1. 自动模式中先由所选 MMS / SpeechBrain 模块检测语言并映射为 SeamlessM4T 语言代码；手动模式跳过检测推理。
2. 设置 `tgt_lang` 为输入语言，生成原文识别文本。
3. 设置 `tgt_lang` 为目标语言，独立生成翻译文本。
4. 两种语言相同时复用原文结果。

两次调用均使用 `generate_speech=False`；本项目不生成翻译语音。音频编码器没有用于强制源语音语言的 `src_lang` 参数，输入语言选项用于指定原文识别输出的语言/文字。[官方用法](https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2)
