# Speech Translation Demo

基于 **SeamlessM4T v2**、**MMS-LID-256 / SpeechBrain** 和 **Gradio** 的语音识别、翻译 Demo。上传音频或使用浏览器麦克风，自动检测输入语言，选择输出语言即可查看识别原文与翻译文本；支持通过公网 HTTPS 链接在其他设备使用。

## 功能

- 音频上传与麦克风录音，默认自动检测输入语言，也可手动选择；提供 21 种常用语言选项。
- 同时显示原文识别、翻译结果和处理耗时。
- 自带中文、英文、法文三个界面示例，无需重新下载示例音频。
- 支持普通 Linux GPU 机器和 Slurm 集群，提供直接分享、登录节点中转及固定域名的启动脚本。

默认限制：单段音频最长 60 秒、文件不超过 25 MB；同一实例按队列顺序处理，等待队列最多 16 个请求。

## 项目结构

```text
Speech_Translation_Demo/
├── app.py                         # 页面、音频处理和模型推理
├── language_detection.py          # MMS GPU / SpeechBrain CPU 语言检测
├── demo_mode.py                   # 演示模式：不加载模型，仅为内置示例返回参考文本
├── ui/                            # 前端设计系统：主题、样式（style.css）与状态动效（app.js）
├── requirements.txt               # Python 依赖
├── .env.example                   # 启动配置模板
├── data/
│   ├── test_samples/              # 自带 MP3 示例及 references.csv
│   └── download_covost2_test_samples.py  # 可选的数据下载工具
├── scripts/
│   ├── run/
│   │   ├── common.sh              # 公共环境初始化
│   │   ├── serve.sh               # 公共启动与进程管理
│   │   ├── serve_share.sh         # 普通 GPU 机器：临时公网链接
│   │   ├── serve_local.sh         # 本地/HTTPS 代理后端
│   │   └── serve_public_tunnel.sh # 普通 GPU 机器：公网服务器中转
│   ├── slurm/
│   │   ├── serve_direct.sh        # 计算节点直接创建公网链接
│   │   ├── serve_login_relay.sh   # 通过登录节点创建公网链接
│   │   └── serve_public_tunnel.sh # 通过公网服务器使用固定域名
│   ├── deploy/
│   │   ├── Caddyfile              # HTTPS 代理配置模板
│   │   ├── run_https_proxy.sh     # 前台运行 HTTPS 入口
│   │   └── speech-demo.service.example # systemd 配置模板
│   ├── check_public_access.py    # 无 GPU 的公网连接检查
│   └── login_relay.py             # 登录节点分享与 SSH 转发
├── docs/
│   ├── environment.md            # 环境、模型及数据准备
│   ├── network.md                # 用直白语言解释公网访问过程
│   ├── deploy_local.md           # 普通 GPU 机器与固定域名部署
│   ├── deploy_slurm.md           # Slurm 三种网络环境的运行步骤
│   ├── configuration.md          # 配置项与应用参数
│   ├── language_detection.md     # 自动语言检测、置信度和适用范围
│   └── faq.md                    # 注意事项、缓存清理与排错
├── ckpts/                        # 本地准备的模型权重
├── .cache/                       # 运行时生成的缓存
└── outs/                         # Slurm 运行日志
```

## 环境与模型部署

需要 **Linux、Python 3.10+、NVIDIA GPU、匹配的 PyTorch CUDA 环境和 FFmpeg**。在普通机器或集群登录节点准备依赖：

```bash
conda create -n st_demo python=3.10 -y
conda activate st_demo
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

已有环境可以直接激活，无需重复安装。默认使用本地目录 `ckpts/seamless-m4t-v2-large/`，目录需要包含完整模型权重、配置和 processor/tokenizer 文件；启动时不会自动下载模型。可以下载官方模型，或通过 `.env.local` 中的 `DEMO_MODEL_DIR` 指定已有路径。

语言检测默认使用 `ckpts/mms-lid-256/`，与 SeamlessM4T 常驻同一张 GPU；MMS 默认 float32，翻译模型默认 float16。原来的 SpeechBrain CPU 检测模块仍可使用，通过 `DEMO_LID_BACKEND=speechbrain` 或 `--lid-backend speechbrain` 切换，默认读取 `ckpts/lang-id-voxlingua107-ecapa/`。只加载所选的检测模块，启动脚本仍申请一张 GPU。模型准备和模块切换见[自动语言检测](docs/language_detection.md)。

完整步骤见[环境与模型准备](docs/environment.md)。依赖安装和模型下载无需运行 GPU 推理；**集群上的模型服务必须通过 Slurm 提交运行**。

## 运行指引

以下命令均从项目根目录执行。首次使用复制配置模板：

```bash
cp -n .env.example .env.local
```

用编辑器修改 `.env.local`：普通机器可以使用已激活的 Python 环境；Slurm 需配置作业中可用的环境初始化方式。脚本会自动读取这个文件，具体选项见[配置说明](docs/configuration.md)。

### 选择启动方式

| 环境 | 运行入口 | 访问方式 |
| --- | --- | --- |
| 普通 GPU 机器可以联网 | `bash scripts/run/serve_share.sh` | 日志中的 `PUBLIC_URL` |
| Slurm 计算节点可以联网 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_direct.sh` | 作业日志中的 `PUBLIC_URL` |
| 计算节点不能联网，登录节点可以 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh` | 登录节点分享进程打印的 `PUBLIC_URL` |
| 普通 GPU 机器有自己的域名和公网入口 | `bash scripts/run/serve_local.sh`，再运行 HTTPS 代理 | 自己的 HTTPS 域名 |
| GPU 机器需经公网服务器使用固定域名 | `bash scripts/run/serve_public_tunnel.sh` | 公网服务器上的 HTTPS 域名 |
| Slurm 需经公网服务器使用固定域名 | `sbatch --partition=GPU_PARTITION scripts/slurm/serve_public_tunnel.sh` | 公网服务器上的 HTTPS 域名 |

`GPU_PARTITION` 要替换为实际 GPU 分区；提交前执行 `mkdir -p outs`。登录节点模式需配置 `DEMO_RELAY_TARGET` 并完成 SSH 登录准备；公网服务器模式需配置 `DEMO_TUNNEL_TARGET` 和 HTTPS 入口。详细可运行步骤分别见 [Slurm 部署](docs/deploy_slurm.md)和[普通机器部署](docs/deploy_local.md)。

Slurm 启动脚本默认申请 **1 张 GPU、4 个 CPU 和每节点 32 GB 主机内存**，为模型加载、音频处理及网页服务预留空间；内存预算包含整个 Demo。MMS 与翻译模型使用同一张可见 GPU。调整资源与查看实际占用见 [Slurm 部署](docs/deploy_slurm.md#cpu-与内存资源)。

### 计算节点不能联网时，为什么仍能在浏览器访问？

模型放在计算节点上；能联网的登录节点帮它建立一个 Gradio 公网地址。浏览器访问该地址时，请求先到 Gradio 的分享服务器，再经登录节点和 SSH 连接送到计算节点；计算节点生成结果后沿这条路返回。

```text
其他设备浏览器
      ↓ HTTPS 公网链接
Gradio 分享服务器
      ↓ 登录节点建立的分享连接
登录节点
      ↓ SSH 内网转发
计算节点上的 Gradio 页面与 GPU 模型
```

**模型没有搬到 Gradio 的公网服务器上**，计算仍在自己的 GPU 上进行。登录节点只负责“把外面的请求送进来”。只看到 `http://0.0.0.0:7860` 时，还没有得到可供其他设备使用的公网地址；应等待 `PUBLIC_URL=https://….gradio.live`。更完整的解释见[公网访问是怎么实现的](docs/network.md)。

### 使用页面与停止服务

1. 打开真实 HTTPS 地址，上传音频或允许浏览器麦克风录音；录音后先停止，等待“音频已就绪”。
2. 输入语言保持“自动检测”，选择目标语言；检测失败或不准确时手动指定输入语言。
3. 点击“识别并翻译”，查看检测语言、置信度、原文和译文；点击示例后同样需要点击按钮。排队和处理期间当前页面的输入控件暂时禁用，完成或出错后恢复。

快捷键 `Ctrl/⌘ + Enter` 可直接提交；顶部“同声传译”页为规划中的功能预告，当前不提供实时传译。

### 无 GPU 预览界面（演示模式）

```bash
python app.py --demo-mode --no-share --host 127.0.0.1
```

演示模式不加载任何模型，页面顶部和结果状态均标注“演示模式”。点击内置示例时返回 CoVoST 2 数据集的参考原文与译文（并非模型输出）；其他音频会提示需在 GPU 环境运行完整模型。也可设置环境变量 `DEMO_MODE=1`。

普通机器前台运行时用 `Ctrl+C` 停止；Slurm 用 `scancel 作业号` 停止。断开登录终端不会结束已提交的 Slurm 作业。

## 注意事项

- 建议用 30 秒以内、清晰且主要使用一种语言的音频。
- 自动检测最多使用 10 秒片段，低置信度时请手动选择语言；默认 MMS 支持粤语类别，SpeechBrain 模式下粤语需手动指定。
- 麦克风录音需要 HTTPS 和浏览器授权。
- Gradio 临时分享链接依赖服务持续运行，通常一周过期；固定地址使用域名部署方案。[分享说明](https://gradio.app/guides/sharing-your-app)
- Slurm 作业受分区/QOS 时限约束，无法仅靠脚本保证无限运行。
- 单个实例使用第一张可见 GPU，多张 GPU 不会自动提高并发。

缓存是否可删除、网络超时、SSH 错误等问题统一见 [Q&A 与排错](docs/faq.md)。

## 模型与参考资料

本项目通过 Transformers 的 `AutoProcessor`、`SeamlessM4Tv2Model` 调用模型，音频转换为单声道、16 kHz，使用 `generate_speech=False` 输出文本。识别和翻译分别设置对应的目标语言，具体行为见[模型与环境说明](docs/environment.md#模型调用方式)。

- [SeamlessM4T v2 官方模型卡](https://huggingface.co/facebook/seamless-m4t-v2-large)：模型文件、语言支持和使用许可。
- [Transformers SeamlessM4T v2 文档](https://huggingface.co/docs/transformers/en/model_doc/seamless_m4t_v2)：模型接口与推理用法。
- [SpeechBrain VoxLingua107 ECAPA 模型](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)：语音语言识别。
- [MMS-LID-256 官方模型卡](https://huggingface.co/facebook/mms-lid-256)：默认语言识别模型、语言列表与使用许可。
- Seamless Communication et al. (2023), [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187).
- [Seamless Communication 官方项目](https://github.com/facebookresearch/seamless_communication)。
- [CoVoST2 示例数据来源](https://huggingface.co/datasets/fixie-ai/covost2)。
