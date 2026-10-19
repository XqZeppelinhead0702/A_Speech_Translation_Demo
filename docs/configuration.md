# 配置与应用参数

[返回 README](../README.md)

## 配置文件

从项目根目录执行：

```bash
cp -n .env.example .env.local
```

所有新启动脚本会自动读取项目根目录的 `.env.local`。它使用 **Bash 赋值语法**：含空格的值加引号，允许 `$HOME` 等 shell 变量。可以用 `DEMO_CONFIG_FILE` 指定另一份完整路径的配置。直接运行 `python app.py` 不会自动读取此文件。

`.env.local` 中的赋值会覆盖同名的已有环境变量；需要临时修改配置时，编辑该文件或使用应用 CLI 参数。命令行参数优先于应用对应的环境变量，但启动脚本会固定本次部署模式需要的 `--share/--no-share` 和监听地址。

## Python 与集群环境

已经激活正确环境的普通机器可以不填写额外初始化配置。Slurm 非交互作业通常需要以下方式之一：

```bash
# 方式一：通过 Conda 初始化脚本激活环境。
DEMO_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
DEMO_CONDA_ENV=st_demo

# 方式二：不激活 Conda，直接指定 Python 与 FFmpeg 路径。
# DEMO_PYTHON=/path/to/env/bin/python
# DEMO_FFMPEG_DIR=/path/to/env/bin
```

集群需要模块时设置 `DEMO_MODULES="模块名1 模块名2"`。若非交互 shell 中尚无 `module` 命令，可通过 `DEMO_SETUP_SH=/path/to/site-setup.sh` 初始化。执行顺序为：本地配置 → 初始化脚本 → module load → Conda 初始化/激活 → 启动服务。

## 正式 Demo 两个页面

正式入口 `app.py` 和 `scripts/run/serve*.sh` 默认启用实时翻译，文件页默认使用 SeamlessM4T 端到端语音翻译；实时页使用 Nemotron → Hy-MT2/vLLM。文件页可选的级联也复用该文本模型服务。首次启动、旧模式回退及验证步骤见[正式 Demo](formal_demo.md)。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEMO_REALTIME` | `1` | 开启同声传译页；CLI `--realtime/--no-realtime` |
| `DEMO_FILE_ASR_TIMEOUT` | `120` | 文件转写等待秒数，Nemotron 在生成步骤/输出回调检查；超时不提交残缺原文给翻译 |
| `DEMO_FILE_TRANSLATION` | `speech` | 默认 `speech`：SeamlessM4T 端到端语音翻译；`text`：默认先用 Nemotron 离线识别，再用文本模型翻译 |
| `DEMO_MT_BACKEND` | `hy-mt2-vllm` | 实时页及文件页可选级联的文本后端；也可 `hy-mt2` / `seamless`，CLI `--mt-backend` |
| `DEMO_MANAGE_VLLM` | `1` | 需要 vLLM 时自动启动和清理引擎；`0` 只连接已有服务 |

## 应用配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEMO_MODEL_DIR` | 项目下 `ckpts/seamless-m4t-v2-large` | 本地完整模型目录 |
| `DEMO_LID_BACKEND` | `mms` | `mms`：GPU MMS-LID-256；`speechbrain`：原来的 CPU 检测器 |
| `DEMO_LID_MODEL_DIR` | 根据模块选择 | `mms` 为 `ckpts/mms-lid-256`，`speechbrain` 为 `ckpts/lang-id-voxlingua107-ecapa` |
| `DEMO_LID_PRECISION` | `float32` | MMS 检测精度：`float32` / `float16` / `bfloat16`；与翻译精度独立，SpeechBrain 不使用此项 |
| `DEMO_LID_MAX_SECONDS` | `10` | 检测片段的时长上限，至少 1 秒 |
| `DEMO_LID_MIN_CONFIDENCE` | `0.5` | 置信度低于此值时提示手动选择，范围 0–1 |
| `DEMO_PYTHON` | `python` | 运行程序的 Python |
| `DEMO_PORT` | `7860` | Gradio 后端端口 |
| `DEMO_HOST` | 脚本 `127.0.0.1` | 仅本地后端脚本可调整；其他模式固定回环地址 |
| `DEMO_PRECISION` | `float16` | `float16` / `bfloat16` / `float32` |
| `DEMO_MAX_SECONDS` | `60` | 音频时长上限 |
| `DEMO_MODE` | `0` | 设为 `1` 不加载模型；文件页音频预览或已有参考文本，实时页明确标注 MOCK，不能衡量质量/时延 |
| `DEMO_STARTUP_TIMEOUT` | `90` | 页面/分享启动超时；不限制模型加载或服务运行时长 |
| `DEMO_ROOT_PATH` | 空 | 域名子路径部署时设置，需与代理配置匹配 |
| `DEMO_AUTH_USER`、`DEMO_AUTH_PASSWORD` | 空 | 同时设置启用 Gradio 登录 |
| `GRADIO_TEMP_DIR` | 每个作业/进程的专属目录 | Gradio 临时文件位置，入口会检查可写性 |
| `DEMO_PROXY_MODE` | `inherit` | 保留现有代理；设 `direct` 可清除本机代理变量 |

例如缩短音频、调整端口：

```bash
bash scripts/run/serve_share.sh --max-seconds 30 --port 7861
# 在 Slurm 上：
sbatch --partition=GPU_PARTITION scripts/slurm/serve_direct.sh --max-seconds 30 --port 7861
```

包装脚本会让 `--port` 同时作用于模型网页和 SSH 转发目标；固定域名代理的后端端口仍需同步修改。更多应用参数用 `python app.py --help` 查看，此操作不会加载模型。

检测配置也可使用 `--lid-backend`、`--lid-model-dir`、`--lid-precision`、`--lid-max-seconds`、`--lid-min-confidence` 参数。默认 MMS 和翻译模型共享第一张可见 GPU，只需一张 GPU；切换到 SpeechBrain 时检测在 CPU 上运行。Slurm 启动脚本默认按分配核数设置 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS`，也可在本地配置中设置更小的值。正式双页 Demo 默认申请 8 个 CPU 和 64 GB 主机内存，调整方法见 [Slurm 资源配置](deploy_slurm.md#cpu-与内存资源)。检测细节见[自动语言检测](language_detection.md)。

## 登录节点中转

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEMO_RELAY_TARGET` | 空，使用中转时必填 | 登录节点 SSH 目标或别名 |
| `DEMO_RELAY_PORT` | `17860` | 登录节点本地转发端口，多作业需区分 |
| `DEMO_RELAY_PROXY_URL` | 空 | 登录节点使用的 HTTP 代理 URL |
| `DEMO_RELAY_FRPC_PROXY_URL` | 未设置，沿用上述代理 | 单独指定 FRPC 分享连接的 HTTP 代理；显式设为空表示 FRPC 直连，HTTPS API 仍使用上述代理 |
| `DEMO_RELAY_SSH_KEY` | 空 | 可选 SSH 私钥路径，不填使用 SSH 默认设置 |
| `DEMO_RELAY_PROJECT_DIR` | 当前项目目录 | 登录节点可读取的项目目录 |
| `DEMO_RELAY_PYTHON` | 当前 Python 的绝对路径 | 登录节点可执行的 Python |

登录节点需要代理时，显式设置 `DEMO_RELAY_PROXY_URL`。如果代理地址使用 `127.0.0.1`，它指登录节点自身，不是计算节点。此变量独立于计算节点的 `DEMO_PROXY_MODE`。

代理能访问 Gradio HTTPS API，不代表能完成分享服务器的 TLS/FRPC 握手。如果已验证登录节点能直接连接分享服务器，而代理路径报 `EOF` / `session shutdown`，可以将两条路径分开：

```bash
DEMO_RELAY_PROXY_URL=http://127.0.0.1:7890
DEMO_RELAY_FRPC_PROXY_URL=
```

第二行必须保留：删除或注释该行会恢复 FRPC 沿用代理的行为。若分享连接也需要代理，则保持未设置，或填写它专用的 HTTP 代理 URL。`7890` 仅为示例，需替换为登录节点实际代理端口。此设置不改变 SSH 中转目标或模型所在节点。

## 实时 ASR 与级联

`DEMO_RELAY_BACKEND_TIMEOUT` 默认为 2400 秒，决定登录节点等待计算节点页面就绪的最长时间，覆盖 vLLM 编译和其他模型加载。该值会通过原 SSH 命令传给登录节点，不改变中转路由、代理或端口；网络预检查和公网分享握手仍保留原超时。

正式实时页使用以下可选配置；文件页级联复用同一个 ASR 与文本翻译器，文件语言检测仍独立。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `REALTIME_ASR_MODEL_DIR` | `ckpts/nemotron-3.5-asr-streaming-0.6b` | 完整本地 ASR 模型 |
| `REALTIME_ASR_PRECISION` | `float32` | Nemotron 推理精度 |
| `REALTIME_ASR_LOOKAHEAD` | `3` | 允许 0/3/6/13；影响右侧上下文等待与识别质量 |
| `REALTIME_TRANSLATION_INTERVAL` | `0.6` | 文本翻译调度间隔秒数，不切音频 |
| `REALTIME_MAX_SECONDS` | `120` | 单次实时输入时长上限 |
| `REALTIME_MT_STREAMING` | `1` | 生成中发布译文候选 |
| `REALTIME_CUDA_STREAMS` | `1` | 进程内为 ASR/MT 分配独立计算流，不保证跨 vLLM 进程抢占 |
| `REALTIME_T2T_MAX_SOURCE_TOKENS` | `1024` | 仅 Seamless 文本后端输入上限 |
| `REALTIME_T2T_MAX_NEW_TOKENS` | `1024` | 仅 Seamless 文本后端输出上限 |

验证步骤与计时说明见[实时页面验证](realtime_testing.md#配置项与同步规则)。麦克风静音结束在浏览器设置：默认开启、2.5 秒，可选 1.5/4 秒或关闭；不使用环境变量，不影响示例完整回放。文件提交时自动试听，可独立停止，不影响识别与翻译。

## 公网服务器中转

`DEMO_TUNNEL_TARGET` 必须设置为公网服务器 SSH 目标/别名；`DEMO_TUNNEL_PORT` 默认为 `17860`，对应 Caddy 在公网服务器上的后端端口。公网服务器模式和登录节点模式由不同入口脚本选择，不会因上次遗留变量误选另一种模式。

`DEMO_PROJECT_DIR` 可指定 Slurm 项目目录；默认使用提交作业时的目录，因此应在项目根目录提交。所有通用 Slurm 脚本默认申请一个节点、一张 GPU，分区、CPU、内存与运行时限通过 `sbatch` 参数按集群要求指定。
## 实时文本翻译后端补充

正式入口默认 `DEMO_MT_BACKEND=hy-mt2-vllm`，通过另一个进程中的 vLLM 服务翻译。服务端默认使用 `DEMO_PYTHON`，无需新建环境；`REALTIME_VLLM_PYTHON` 仅为可选 Python 覆盖项。`REALTIME_VLLM_MODEL_DIR`、`REALTIME_VLLM_API_BASE`、`REALTIME_VLLM_SERVED_MODEL` 配置模型和连接。CLI 可用 `--mt-model-dir`、`--mt-api-base`、`--mt-served-model`。资源/超时参数见 [vLLM 部署](vllm.md#资源与性能配置)。此后端不受本进程的 `REALTIME_CUDA_STREAMS` 翻译队列设置控制。

私有通用测试入口默认 `REALTIME_MT_BACKEND=seamless`；正式入口优先使用 CLI `--mt-backend`，否则读 `DEMO_MT_BACKEND`，默认 `hy-mt2-vllm`，不受旧测试变量 `REALTIME_MT_BACKEND` 影响。设置 `hy-mt2` 可使用原始 Hy-MT2-1.8B/7B 权重。其独立配置为 `REALTIME_HY_MT_MODEL_DIR`（默认 `ckpts/Hy-MT2-1.8B`）、`REALTIME_HY_MT_PRECISION`（默认 `bfloat16`）、`REALTIME_HY_MT_MAX_SOURCE_TOKENS`（默认 4096，包含模板）、`REALTIME_HY_MT_MAX_NEW_TOKENS`（默认 1024）及 `REALTIME_HY_MT_SAMPLING`（默认 0，1 为官方推荐采样）。不沿用 `DEMO_PRECISION` 或 Seamless 的 token 上限。

正式入口接受 `--mt-backend`、`--mt-model-dir`、`--mt-precision`、`--mt-max-source-tokens`、`--mt-max-new-tokens`、`--mt-sampling`，显式 CLI 参数优先于环境变量，并应用到两页共享的文本翻译器。详细步骤见 [Hy-MT2 对照](hy_mt_testing.md)。

实时 Hy-MT2 默认启用 `REALTIME_MT_STREAMING=1`，在推理生成时发布独立草稿；设置 0 可对比原先完整译文模式。Seamless 不支持此回调时自动沿用完整输出。草稿字段和协作者接入见 [生成中译文协议](realtime_development.md#hy-mt2-生成中译文)。

`REALTIME_CUDA_STREAMS=1` 默认为 ASR 和文本翻译创建独立计算流，ASR 请求高优先级；0 为原队列对照模式。此配置仅在 GPU 作业内加载模型时使用，CPU mock 不初始化 CUDA。样本播放节奏在测试页诊断区选择，默认实时连续播放，不等待原文或译文；网络适配模式只在上传积压时暂停。
