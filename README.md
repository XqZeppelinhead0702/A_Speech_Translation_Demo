# Speech Translation Demo

基于 Gradio 的多语言语音识别与翻译 Demo，提供**文件/录音翻译**和**同声传译**两个页面，可通过公网 HTTPS 链接在其他设备使用。

- **文件/录音翻译**：默认 SeamlessM4T v2 端到端语音翻译，直接从音频生成译文；另行转写音频以显示原文。默认自动检测输入语言，提供 21 种常用语言，也可手动指定。通过模型下拉框可选 Nemotron 非流式识别 → Hy-MT2/vLLM 级联翻译。提交时自动播放输入音频，试听可单独暂停或停止。
- **同声传译**：Nemotron 3.5 连续流式 ASR → Hy-MT2/vLLM。手动选择说话语言，原文先更新、译文跟进；支持持续录音和带声音的示例回放。麦克风默认在讲话后静音 2.5 秒自动结束并补齐最终结果，也可调整等待时间或关闭，详见[页面使用说明](docs/formal_demo.md#演示播放与录音自动结束)。
- 实时页和文件页的可选级联共用一份 Hy-MT2 服务，不重复加载翻译权重；文件任务与实时会话互斥，避免争抢模型。原生 Transformers Hy-MT2、Seamless 文本后端仍可选择。
- 内置 10 条约 15–27 秒的真人朗读，覆盖 8 种输入语言，音频随项目提供，无需重新下载。

## 项目结构

```text
Speech_Translation_Demo/
├── app.py                         # 正式两页界面与文件/录音处理
├── language_detection.py          # MMS GPU / SpeechBrain CPU 语言检测
├── demo_mode.py                   # 无 GPU 的界面预览
├── sample_catalog.py              # 两页共用的样本清单
├── realtime/                      # 流式 ASR、文本翻译、HTTP/WS 接口与引擎管理
├── ui/                            # 两页主题、样式、录音/回放与字幕显示
├── requirements.txt               # 基础 Python 依赖
├── requirements-streaming.txt     # 实时音频与 WebSocket 依赖
├── requirements-vllm.txt          # 默认两页 Demo 的完整依赖入口
├── .env.example                   # 启动配置模板
├── data/
│   └── test_samples/              # 展示 MP3、样本清单和来源说明
├── scripts/
│   ├── run/                       # 普通 GPU 机器启动与共用环境初始化
│   │   ├── serve_share.sh         # Gradio 临时公网链接
│   │   ├── serve_local.sh         # 本机 / HTTPS 代理后端
│   │   ├── serve_public_tunnel.sh # 公网服务器中转
│   │   └── serve_vllm.sh          # 单独启动 Hy-MT2 服务
│   ├── slurm/
│   │   ├── serve_direct.sh        # 计算节点直连公网
│   │   ├── serve_login_relay.sh   # 登录节点中转
│   │   ├── serve_public_tunnel.sh # 公网服务器中转
│   │   ├── check_vllm.sh          # 不加载权重的环境检查
│   │   └── serve_vllm.sh          # 独立引擎作业
│   ├── deploy/                    # Caddy / systemd 部署模板
│   ├── check_public_access.py     # 公网网络检查
│   ├── check_vllm.py              # 配置、tokenizer 和版本检查
│   └── login_relay.py             # 登录节点分享与 SSH 转发
├── docs/
│   ├── environment.md            # 环境和模型准备
│   ├── formal_demo.md            # 正式两页 Demo 的启动、验证和回退
│   ├── architecture.md           # 两种翻译路径、模型共享和模块分工
│   ├── network.md                # 公网访问原理
│   ├── deploy_slurm.md           # Slurm 三种网络环境的运行步骤
│   ├── deploy_local.md           # 普通机器与固定域名部署
│   ├── configuration.md          # 配置项与应用参数
│   ├── language_detection.md     # 自动语言检测
│   ├── vllm.md                   # vLLM 版本、兼容性和性能配置
│   ├── realtime_development.md   # 实时接口与协作者接入
│   ├── realtime_testing.md       # 正式页面验证与延迟诊断
│   ├── hy_mt_testing.md          # Hy-MT2 1.8B / 7B 配置与效果对照
│   └── faq.md                    # 缓存、网络与常见问题
├── ckpts/                        # 本地准备的模型权重
├── .cache/                       # 运行缓存
└── outs/                         # 作业与引擎日志
```

## 环境与模型部署

需要 Linux、Python 3.10+、NVIDIA GPU、匹配的 PyTorch CUDA 环境和 FFmpeg。现有 `st_demo` 已完成 vLLM 实时测试时，直接沿用，不需要重新建环境。当前测试组合为 Torch/Torchaudio 2.8.0、Transformers 5.18.0、Gradio 6.29.1、vLLM 0.10.2；不代表所有驱动和硬件组合都兼容。

从零安装时参照[环境与模型准备](docs/environment.md)：先安装匹配驱动的 Torch/Torchaudio CUDA 构建，再用 `python -m pip install -r requirements-vllm.txt` 安装默认两页 Demo 的依赖。该清单包含基础、实时依赖和 vLLM 0.10.2；选择原生 Hy-MT2 或仅文件翻译时，可使用对应精简依赖。启动脚本不自动安装、升级或修改包；现有环境无需重复执行安装命令。

默认准备以下完整模型目录，保留权重、配置和 processor/tokenizer 等配套文件：

```text
ckpts/
├── seamless-m4t-v2-large/          # 文件转写与原直接语音翻译
├── mms-lid-256/                    # 文件输入语言检测
├── nemotron-3.5-asr-streaming-0.6b/ # 实时及文件级联识别
└── Hy-MT2-1.8B/                    # 两页共用文本翻译，也可使用 7B
```

运行时使用本地文件，不自动下载模型。目录可通过 `.env.local` 指定；切换 `DEMO_LID_BACKEND=speechbrain` 可使用原 CPU 检测器，需准备 `ckpts/lang-id-voxlingua107-ecapa/`。下载步骤、模型路径和依赖细节见[环境说明](docs/environment.md)及[vLLM 兼容说明](docs/vllm.md)。

通用正式 Slurm 脚本默认申请 **1 GPU、8 CPU、64 GB 主机内存**。建议先用 A100 80 GB 做整体测试；默认 vLLM 按整卡显存的 40% 制定预算，其余模型也需显存。集成后的总峰值仍以实际作业为准，说明见[正式 Demo](docs/formal_demo.md#资源说明)。

## 运行指引

以下命令均从项目根目录执行。首次使用复制配置模板：

```bash
cp -n .env.example .env.local
mkdir -p outs
```

在 `.env.local` 中填写作业可用的 Python/Conda、FFmpeg 以及对应网络模式的配置。已有配置可继续使用，无需覆盖；默认文件页使用端到端语音翻译，实时页使用 Hy-MT2/vLLM，程序自动启动一份文本翻译引擎。直接运行 `python app.py` 不自动读取此文件，建议使用下列脚本。

| 环境 | 启动命令 | 浏览器入口 |
| --- | --- | --- |
| 普通 GPU 机器可以联网 | `bash scripts/run/serve_share.sh` | 日志中的 `PUBLIC_URL` |
| Slurm 计算节点可以联网 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_direct.sh` | 日志中的 `PUBLIC_URL` |
| 计算节点不能联网，登录节点可以 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh` | 日志中的 `PUBLIC_URL` |
| 普通 GPU 机器已有 HTTPS 代理 | `bash scripts/run/serve_local.sh` | 自己配置的 HTTPS 域名 |
| GPU 机器经公网服务器使用固定域名 | `bash scripts/run/serve_public_tunnel.sh` | 公网服务器的 HTTPS 域名 |
| Slurm 经公网服务器使用固定域名 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_public_tunnel.sh` | 公网服务器的 HTTPS 域名 |

`GPU_PARTITION` 替换为实际分区。登录节点模式需设置 `DEMO_RELAY_TARGET` 并准备 SSH 连接；公网服务器模式需设置 `DEMO_TUNNEL_TARGET` 和 HTTPS 入口。详细可运行步骤见 [Slurm 部署](docs/deploy_slurm.md)、[普通机器部署](docs/deploy_local.md)和[配置说明](docs/configuration.md)。

启动顺序为：网络预检查 → vLLM 就绪 → 加载其他模型 → 开放页面。首次编译可能较慢，引擎日志在 `outs/vllm-engine-作业号.log`，正式应用日志在 `outs/demo-作业号.out/.err`。两页使用同一个 URL，无需对公网开放 vLLM 端口。

### 公网访问是怎么实现的

计算节点不联网时，登录节点帮它建立 Gradio 公网分享连接。外部请求按下面的路线进入，计算结果沿原路返回：

```text
其他设备浏览器 → Gradio 公网分享服务器 → 登录节点 → SSH 内网转发 → 计算节点上的 Demo
```

模型始终在自己的 GPU 上，登录节点负责转发，Gradio 分享服务器提供外部入口。看到 `http://0.0.0.0:7860` 只是本地监听地址，应等待 `PUBLIC_URL=https://….gradio.live`，再用其他设备打开这个 HTTPS 链接。更完整的解释见[网络原理](docs/network.md)。

### 使用与停止

页面直接显示“语音翻译”和“同声传译”两个切换按钮，窄屏也不会把模式藏进下拉菜单。

1. 在“语音翻译”页上传音频或录音；录音后先停止，等“音频已就绪”。选择语言和翻译方式，点击“识别并翻译”。示例按钮会填入音频及语言，仍需提交。`Ctrl/⌘ + Enter` 可提交当前可见文件页。
2. 在“同声传译”页手动选择输入与输出语言，开始录音，或播放内置示例。原文先更新，译文独立跟进；示例声音持续播放，不为等待字幕暂停。麦克风讲话后静音 2.5 秒默认自动停止，也可手动停止；随后等待最终结果，或点击取消。
3. 用同一样本对照两个页面及文件页的原直接语音翻译选项。完整验证顺序、切换 7B 和连接已有引擎见[正式 Demo](docs/formal_demo.md)。

普通机器用 `Ctrl+C` 停止，Slurm 用 `scancel 作业号`，托管 vLLM 会一起退出。断开登录终端不会结束已提交的 Slurm 作业。

仅运行原文件/录音直接语音翻译、不加载实时 ASR 或 Hy-MT2 时：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh \
  --no-realtime --file-translation speech
```

仅预览界面、不加载 GPU 模型时：

```bash
bash scripts/run/serve_local.sh --demo-mode
```

文件页仅提供音频预览或已有参考文本；实时页显示明确标注的 MOCK 占位字幕，不能用于评估识别、翻译质量或模型延迟。

## 注意事项与常见问题

- 文件最长 60 秒、25 MB；实时输入默认最长 120 秒。两页同时只处理一个模型任务；实时会话进行中提交文件会提示稍后重试。
- 浏览器麦克风需要 HTTPS（localhost 除外）和授权。公网 URL 必须等待分享进程成功后获取。
- 文件语言检测最多使用 10 秒音频，适合主要使用一种语言的清晰输入；置信度不足时请手动指定。实时页暂不自动检测语言。
- 实时译文仍可能随上下文修订，尚未按语义块永久锁定前文。较长会话仍可能增加翻译延迟，可展开“延迟诊断”查看阶段计时。
- [架构说明](docs/architecture.md)介绍模型共享与两页处理路径；协作开发可查看[实时接口与同步协议](docs/realtime_development.md)。
- 缓存能否删除、上传失败、DNS/SSH/分享连接和显存错误，统一见 [Q&A](docs/faq.md)。样本来源与裁剪信息见[音频来源](data/test_samples/SOURCES.md)。

## 引用与资料

- [SeamlessM4T v2 模型卡](https://huggingface.co/facebook/seamless-m4t-v2-large)、[Transformers 文档](https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2)；Seamless Communication et al. (2023), [Seamless](https://arxiv.org/abs/2312.05187)。
- [Nemotron 3.5 流式 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)。
- [Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B)。
- [MMS-LID-256](https://huggingface.co/facebook/mms-lid-256)、[SpeechBrain VoxLingua107](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)。
- [CoVoST2 旧样本来源](https://huggingface.co/datasets/fixie-ai/covost2)、[LibriVox 展示样本来源](data/test_samples/SOURCES.md)。

模型和数据的使用条件以各自官方说明为准。
