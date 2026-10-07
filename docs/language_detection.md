# 自动语音语言检测

[返回 README](../README.md) · [配置项](configuration.md)

## 使用方式与检测模块

上传音频或录音，输入语言保持“自动检测”，选择目标语言并点击“识别并翻译”。右侧显示使用的检测模型、语言、置信度、原文和译文；状态中显示检测耗时。需要纠正时手动选择实际输入语言再提交；手动模式跳过检测推理。点击“清空”恢复自动检测。

支持两个检测模块，**默认使用 MMS-LID-256**，原来的 SpeechBrain 模块及权重均保留：

| 模块 | 启动配置 | 运行设备 | 默认目录 | 当前 Demo 可映射的语音语言 |
| --- | --- | --- | --- | --- |
| MMS-LID-256 | `DEMO_LID_BACKEND=mms` | 与翻译模型相同的 GPU | `ckpts/mms-lid-256` | 20 种，含普通话和粤语 |
| SpeechBrain VoxLingua107 | `DEMO_LID_BACKEND=speechbrain` | CPU | `ckpts/lang-id-voxlingua107-ecapa` | 19 种，粤语需手动选择 |

仅在服务启动时加载所选检测模型，之后复用；切换模块需要重启作业，不在页面中切换。默认模式在一张 GPU 上常驻 MMS 与 SeamlessM4T 两个模型；选择 SpeechBrain 时不加载 MMS。

## 配置与启动

通用启动脚本读取 `.env.local`，默认配置：

```bash
DEMO_LID_BACKEND=mms
DEMO_LID_PRECISION=float32
DEMO_LID_MAX_SECONDS=10
DEMO_LID_MIN_CONFIDENCE=0.5
# 只有模型在其他位置时才设置，且必须与模块匹配。
# DEMO_LID_MODEL_DIR=/path/to/mms-lid-256
```

然后使用原来的启动脚本，例如登录节点中转：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh
```

`GPU_PARTITION` 替换为实际 GPU 分区，SSH 和环境配置仍按 [Slurm 部署](deploy_slurm.md)准备。

切回原来的 CPU 检测器，可以修改 `DEMO_LID_BACKEND=speechbrain`，或使用 CLI：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh --lid-backend speechbrain
```

未指定模型路径时，程序根据模块选择对应目录。若旧配置中显式设置了 `DEMO_LID_MODEL_DIR`，切换模块时也要修改或删除该设置；CLI `--lid-model-dir` 可覆盖它。已有本地私有启动脚本直接运行 `app.py` 时，同样可在 `sbatch` 后的脚本参数中传入以上应用参数；直接运行 Python 不自动读取 `.env.local`。

## 本地权重与环境

MMS 使用 [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) 的**语言分类权重**，不是 MMS 的语音转写权重。默认目录：

```text
ckpts/mms-lid-256/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

也支持 `pytorch_model.bin` 或官方分片权重及完整索引文件。目录同时有 safetensors 和 bin 时优先读取 safetensors，不会将两份重复权重都加载。已有完整下载无需重下。MMS 使用项目现有的 Transformers、PyTorch 依赖，不新增其他推理框架。

SpeechBrain 目录保持原样：

```text
ckpts/lang-id-voxlingua107-ecapa/
├── hyperparams.yaml
├── embedding_model.ckpt
├── classifier.ckpt
└── label_encoder.txt
```

依赖按项目 `requirements.txt` 安装。两个模块都只读取本地文件；SpeechBrain 官方 YAML 中的下载来源在加载时覆盖为本地路径，下载文件本身不修改。

在新机器仅下载 MMS 权重时，可在联网位置执行，再复制到计算节点可读的目录；此命令不运行 GPU 模型：

```bash
python - <<'DOWNLOAD_LID'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/mms-lid-256",
    local_dir="ckpts/mms-lid-256",
    allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "*.safetensors.index.json", "langs.txt"],
)
DOWNLOAD_LID
```

## 处理流程、设备与精度

```text
音频校验 → 单声道 / 16 kHz → 去除首尾低能量部分
                           ↓
               最多 10 秒片段送入所选语言分类器
                           ↓
               最高类别、置信度与 Seamless 代码映射
                           ↓
                完整音频生成原文和翻译
```

首尾裁剪按音量进行，不是独立的语音/音乐分类器。检测至少需要 1 秒有效音频；检测片段限长不改变完整音频的识别和翻译。类别在 MMS 的完整 256 类或 SpeechBrain 的完整 107 类上判断，不强行限定到 Demo 的语言选项。

MMS 使用 `AutoFeatureExtractor`、`Wav2Vec2ForSequenceClassification`，在 `cuda:0` 上推理。`cuda:0` 指第一张可见 GPU；Slurm 屏蔽其他 GPU 时，仍使用作业分配的那一张。翻译模型也使用同一张可见 GPU，队列与模型锁保证检测和翻译串行执行。

MMS 默认 `float32`，独立于 SeamlessM4T 默认 `float16` 的精度设置。可用 `DEMO_LID_PRECISION` 或 `--lid-precision` 选择 `float16` / `bfloat16`；较低精度会影响显存、速度及数值结果，需要实际验证。`bfloat16` 会检查 GPU 支持情况；SpeechBrain 不使用该精度选项。

本地 MMS 权重含约 9.66 亿参数，FP32 权重约 3.6 GiB；这只是权重，不包含推理中间结果、翻译模型和 CUDA 缓存。脚本仍申请 **1 张 GPU、4 个 CPU、32 GB 主机内存**。尚未运行双模型 GPU 推理测量实际峰值或检测耗时，需以 Slurm 作业实测为准。延迟随片段长度、GPU、精度和节点负载变化，页面显示单次检测耗时。

## 语言范围与失败处理

两种模块都可映射普通话、英语、法语、德语、西班牙语、日语、韩语、俄语、葡萄牙语、意大利语、阿拉伯语、印地语、泰语、越南语、印尼语、土耳其语、荷兰语、波兰语、乌克兰语；MMS 额外映射粤语。

- MMS 标签主要是三字母代码，其中 `ara` 映射为 SeamlessM4T 的 `arb`；SpeechBrain 的两字母标签使用独立映射。
- 检测到未映射的语言时提示确认，不强行改成已支持语言。
- 默认最高类别置信度低于 0.5 时停止本次推理，提示手动选择或提供更清晰片段。
- MMS 置信度由 logits 的 softmax 得到；SpeechBrain 对对数后验取指数。分数不是经过校准的正确率，也不能直接用来比较两个模型的准确率；高分仍可能误判。
- MMS 有独立普通话、粤语类别；SpeechBrain 无独立粤语类别，粤语请手动指定。
- 简体/繁体是文字输出设置，不能从声音区分；自动普通话映射为 `cmn`，需要繁体原文时手动选择对应选项。
- 每段音频返回一个主要语言类别，不进行逐段多语言检测；混合语言音频需手动确认。

## API 返回值与引用

Gradio `/translate` 仍接受音频、输入语言、目标语言，自动模式使用 `auto`；依次返回原文、译文、输入语言识别说明、状态，共四项。

- [MMS-LID-256 模型卡](https://huggingface.co/facebook/mms-lid-256)：官方调用示例、语言范围及 CC-BY-NC 4.0 使用许可。
- Pratap et al. (2023), [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516)。
- [SpeechBrain VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)：原来的 CPU 检测模型。
