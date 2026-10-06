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

## 应用配置

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEMO_MODEL_DIR` | 项目下 `ckpts/seamless-m4t-v2-large` | 本地完整模型目录 |
| `DEMO_PYTHON` | `python` | 运行程序的 Python |
| `DEMO_PORT` | `7860` | Gradio 后端端口 |
| `DEMO_HOST` | 脚本 `127.0.0.1` | 仅本地后端脚本可调整；其他模式固定回环地址 |
| `DEMO_PRECISION` | `float16` | `float16` / `bfloat16` / `float32` |
| `DEMO_MAX_SECONDS` | `60` | 音频时长上限 |
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

## 登录节点中转

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEMO_RELAY_TARGET` | 空，使用中转时必填 | 登录节点 SSH 目标或别名 |
| `DEMO_RELAY_PORT` | `17860` | 登录节点本地转发端口，多作业需区分 |
| `DEMO_RELAY_PROXY_URL` | 空 | 登录节点使用的 HTTP 代理 URL |
| `DEMO_RELAY_SSH_KEY` | 空 | 可选 SSH 私钥路径，不填使用 SSH 默认设置 |
| `DEMO_RELAY_PROJECT_DIR` | 当前项目目录 | 登录节点可读取的项目目录 |
| `DEMO_RELAY_PYTHON` | 当前 Python 的绝对路径 | 登录节点可执行的 Python |

登录节点需要代理时，显式设置 `DEMO_RELAY_PROXY_URL`。如果代理地址使用 `127.0.0.1`，它指登录节点自身，不是计算节点。此变量独立于计算节点的 `DEMO_PROXY_MODE`。

## 公网服务器中转

`DEMO_TUNNEL_TARGET` 必须设置为公网服务器 SSH 目标/别名；`DEMO_TUNNEL_PORT` 默认为 `17860`，对应 Caddy 在公网服务器上的后端端口。公网服务器模式和登录节点模式由不同入口脚本选择，不会因上次遗留变量误选另一种模式。

`DEMO_PROJECT_DIR` 可指定 Slurm 项目目录；默认使用提交作业时的目录，因此应在项目根目录提交。所有通用 Slurm 脚本默认申请一个节点、一张 GPU，分区、CPU、内存与运行时限通过 `sbatch` 参数按集群要求指定。
