# Slurm 部署

[返回 README](../README.md) · [网络原理](network.md) · [配置项](configuration.md)

所有命令从项目根目录执行。模型只能在分配了 GPU 的作业中运行；登录节点可以安装依赖、准备文件和做不使用 GPU 的网络检查。

## 共同准备

1. 按[环境说明](environment.md)准备 Python 环境、FFmpeg、模型和示例。
2. 创建配置：`cp -n .env.example .env.local`，填写作业中有效的环境初始化方式。
3. 执行 `mkdir -p outs`，日志目录必须在提交前存在。
4. 确认 GPU 分区、CPU/内存和时限；将下文 `GPU_PARTITION` 替换为实际分区。

新脚本不写死集群模块、账号、分区或共享目录。默认申请一节点、一任务、一 GPU、8 个 CPU 和每节点 64 GB 主机内存；时限等参数以所在集群要求为准。

## 从旧的 serve_demo.sh 迁移

旧的 `scripts/slurm/serve_demo.sh` 将特定集群的环境和网络设置写在脚本中；公开版本将环境配置移到 `.env.local`，按网络情况提供三个入口：

| 旧脚本使用的方式 | 新入口 |
| --- | --- |
| 设置 `DEMO_RELAY_TARGET`，通过登录节点生成 Gradio 公网链接 | `scripts/slurm/serve_login_relay.sh` |
| 不使用中转，计算节点自己生成 Gradio 公网链接 | `scripts/slurm/serve_direct.sh` |
| 设置 `DEMO_TUNNEL_TARGET`，经公网服务器访问固定域名 | `scripts/slurm/serve_public_tunnel.sh` |

如果旧脚本已通过登录节点中转成功运行，选择 `serve_login_relay.sh`。把原来的环境初始化与中转设置移入项目根目录的 `.env.local`，例如：

```bash
# 替换为旧脚本中实际使用的模块和路径；不需要模块时省略该项。
DEMO_MODULES="ANACONDA_MODULE CUDA_MODULE GCC_MODULE"
DEMO_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
DEMO_CONDA_ENV=st_demo
DEMO_FFMPEG_DIR=/path/to/ffmpeg/bin

DEMO_RELAY_TARGET=YOUR_USER@LOGIN_HOST
DEMO_RELAY_PORT=17860
# 登录节点需要代理时填写；不需要则保持为空。
DEMO_RELAY_PROXY_URL=

DEMO_PORT=7860
DEMO_LID_BACKEND=mms
DEMO_LID_PRECISION=float32
```

登录节点的项目和 Python 默认使用计算节点上的相同路径；不同路径时设置 `DEMO_RELAY_PROJECT_DIR` 和 `DEMO_RELAY_PYTHON`。保留已经验证可用的 SSH 密钥与 `known_hosts`，无需因更换入口重新创建。其他语言检测配置可沿用 `.env.example` 的默认值。

原来的 `#SBATCH --partition=...` 改为提交时指定，**分区不能通过 `.env.local` 设置**，因为 Slurm 在运行脚本、读取该文件之前就已完成资源分配：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh
```

不要同时提交旧入口和新入口来迁移同一个服务；它们可能争用相同的登录节点中转端口。先停止旧作业，再提交新作业。新日志位置为 `outs/demo-JOB_ID.out`、`outs/demo-JOB_ID.err`，不再使用旧脚本的固定 `outs/demo.out` / `outs/demo.err`。

新入口会自动读取 `.env.local`；旧的 `serve_demo.sh` 不会自动读取该文件。应用及中转代码仍由两类入口共用，但配置来源不同。

## CPU 与内存资源

三个公开 Slurm 启动脚本已包含：

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

正式入口现在默认加载 Nemotron 流式 ASR、SeamlessM4T、MMS-LID-256，并在同一作业中启动一份 Hy-MT2/vLLM 服务；仍只申请一张 GPU。默认 64 GB 是整个服务的主机内存预算，GPU 显存由分配的 GPU 决定，不能通过增加 `--mem` 扩大显存。建议先用 A100 80 GB 验证整体运行；本次集成尚未测量四个模型的总显存峰值。

原 Python、SSH、代理、中转参数继续使用，两页使用同一个公网 URL。启动先等待 vLLM，再加载其他模型，`scancel` 时清理托管引擎；首次验证、后端切换及原模式回退见[正式 Demo](formal_demo.md)。

## A. 计算节点可联网：直接分享

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_direct.sh
```

这个入口在实际计算节点先检查 Gradio API 和分享服务器，检查成功才启动模型。它不使用登录节点中转。模型与分享连接建立后，日志会打印 `PUBLIC_URL=https://….gradio.live`。

如果提交环境里的代理只在登录节点有效，可在 `.env.local` 设置 `DEMO_PROXY_MODE=direct`，让计算节点直接尝试联网；如果节点确实不能联网，使用下面的方案 B。

## B. 计算节点不能联网：登录节点中转

### 1. 配置目标与环境

在 `.env.local` 设置：

```bash
DEMO_RELAY_TARGET=YOUR_USER@LOGIN_HOST
DEMO_RELAY_PORT=17860
DEMO_RELAY_PROXY_URL=
```

将 SSH 账号和目标替换为实际值。若登录节点需要代理，填写它使用的 HTTP 代理 URL。登录节点和计算节点通常共享 home/项目/Conda；若路径不同，再设置 `DEMO_RELAY_PROJECT_DIR`、`DEMO_RELAY_PYTHON`。

如果 API 通过代理可用，但 FRPC 分享握手报 `EOF` / `session shutdown`，且分享服务器直连已验证可用，可保留 `DEMO_RELAY_PROXY_URL` 并显式添加 `DEMO_RELAY_FRPC_PROXY_URL=`，让 API 走代理、FRPC 直连。完整诊断见 [FRPC 握手排错](faq.md#frpc-报-login-to-server-failed-eof--session-shutdown)。

### 2. 准备免交互 SSH

先在集群运行账号下首次连接，核对管理员提供的主机指纹后建立 `known_hosts` 记录，再验证作业不需要输入密码：

```bash
ssh YOUR_USER@LOGIN_HOST
# 登录成功后退出，再执行检查：
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes YOUR_USER@LOGIN_HOST true
```

需要专用密钥时，可在没有同名文件的前提下创建并安装公钥：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/speech_demo_relay -N ''
ssh-copy-id -i ~/.ssh/speech_demo_relay.pub YOUR_USER@LOGIN_HOST
ssh -i ~/.ssh/speech_demo_relay -o BatchMode=yes -o StrictHostKeyChecking=yes YOUR_USER@LOGIN_HOST true
```

随后在 `.env.local` 加入 `DEMO_RELAY_SSH_KEY="$HOME/.ssh/speech_demo_relay"`。如果默认密钥已经可用，不必创建新密钥。集群须允许计算节点 SSH 到登录节点及 remote TCP forwarding，并允许登录节点承担网络中转。

### 3. 可选：先做 CPU 检查

准备相同环境和变量后，可以检查实际 SSH 会话中的网络，不加载模型：

```bash
# 加载本地配置与环境，只用于本次检查。
source scripts/run/common.sh
python_executable="$DEMO_PYTHON"
"$python_executable" -u scripts/login_relay.py connect \
  --target "$DEMO_RELAY_TARGET" --proxy-url "${DEMO_RELAY_PROXY_URL:-}" \
  --python "${DEMO_RELAY_PYTHON:-$(command -v "$python_executable")}" \
  --project-dir "${DEMO_RELAY_PROJECT_DIR:-$PWD}" \
  --check-only --test-tunnel
```

检查建立的诊断隧道会立即关闭，不能把它当作 Demo 使用。登录节点上的检查无法代替实际计算节点的内网连通性验证；作业启动时还会再次检查。

### 4. 提交并访问

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh
```

脚本会先验证 SSH 会话中的公网连接，再启动 GPU 模型和中转；计算节点不连接 Gradio 公网服务。等待作业日志打印实际 `PUBLIC_URL` 后，在其他设备浏览器打开。

## C. 公网服务器与固定域名

先按[固定域名部署](deploy_local.md#公网服务器中转)准备公网服务器、HTTPS 入口和 SSH。然后在 `.env.local` 设置：

```bash
DEMO_TUNNEL_TARGET=speech-demo-public
DEMO_TUNNEL_PORT=17860
```

`speech-demo-public` 为自己在 `~/.ssh/config` 中配置的别名。计算节点需能 SSH 到该公网服务器，不需要直接连接 Gradio 分享服务。

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_public_tunnel.sh
```

模型加载完成后访问自己的 HTTPS 域名。该入口不生成 Gradio 分享 URL；域名由公网服务器的 Caddy 提供。

## 查看日志与停止

假设返回作业号 `123456`：

```bash
squeue -j 123456
tail -f outs/demo-123456.out outs/demo-123456.err
scancel 123456
```

标准输出包含 URL；错误日志包含模型加载与启动错误。三种脚本都将服务绑定在作业生命周期内；中转或后端任一方退出会结束该服务组合并释放 GPU。

## 持续运行与时限

应用没有主动退出计时器，但 Slurm 分区/QOS 的时限、抢占和节点故障仍可能结束作业。查看实际限制：

```bash
scontrol show partition GPU_PARTITION
scontrol show job 123456
```

脚本没有固定 `--time`，使用集群默认时限。若分区允许，可通过 `sbatch --time=0 ...` 请求无限运行；不允许时需申请适合长期服务的分区/配置，不能由脚本绕过。新脚本不自动重新提交作业。[Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)
