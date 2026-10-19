# Hy-MT2 的 vLLM 部署

[返回 README](../README.md) · [环境准备](environment.md) · [正式 Demo](formal_demo.md) · [Hy-MT2 对照](hy_mt_testing.md)

## 接入与版本

正式同声传译页默认使用 `hy-mt2-vllm`；文件页选择级联时复用同一引擎。Demo 进程加载 tokenizer，通过本机 `/v1/completions` 接收流式译文，不再加载一份 Hy-MT2 权重。默认引擎与 ASR 在同一作业、同一 GPU、同一 Python 环境中运行，vLLM 仅绑定 `127.0.0.1:8000`，浏览器无需访问或转发此端口。

| 组件 | 当前基线 |
| --- | --- |
| vLLM | 0.10.2 |
| Torch / Torchaudio | 2.8.0，CUDA 12.8 wheel |
| Transformers | 5.18.0 |
| Python | 3.10 |
| 模型 | 原始 Hy-MT2-1.8B / 7B dense 权重 |

这是项目现有可用组合，不是最低版本要求。[Hy-MT2 官方](https://huggingface.co/tencent/Hy-MT2-1.8B#inference-and-deployment)提供 vLLM 源码安装示例，没有承诺最低版本。[vLLM 0.10.2 GPU 安装说明](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html)说明其二进制与 CUDA/PyTorch 的关系；显卡型号本身不足以证明所有驱动和 kernel 兼容。

新环境可使用 `requirements-vllm.txt`；已经运行成功的环境直接沿用，不自动安装或修改包。其他引擎版本需分别核对结构注册、配置/tokenizer 与 CUDA kernel，不能仅按版本号大小判断。

### tokenizer 兼容层

vLLM 0.10.2 会读取旧属性 `all_special_tokens_extended`，而当前 Transformers 5 已删除它。`scripts/vllm_compat/sitecustomize.py` 仅在 vLLM 0.10.2 + Transformers 5 且属性缺失时恢复该接口。

启动脚本仅为引擎及其子进程加入此兼容层，不修改 `site-packages`，也不改变 ASR/Demo 进程的导入行为。`REALTIME_VLLM_TOKENIZER_COMPAT=0` 可关闭进行诊断；它不是任意版本组合的通用兼容补丁。

## 默认托管运行

在项目根目录配置 `.env.local`，按网络环境选择正式入口，例如：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh
```

默认相当于 `--mt-backend hy-mt2-vllm --manage-vllm`。先启动引擎并等待 `/health`，再加载其他模型并开放两页。首次编译默认最多等待 1800 秒，进度和错误写入 `outs/vllm-engine-作业号.log`；登录节点默认等后端 2400 秒。

客户端在加载时验证 `/v1/models`、`/tokenize` 与可读取的上下文上限。引擎和客户端必须使用相同模型/tokenizer。引擎异常退出会结束托管 Demo；`scancel` 或普通机器的 `Ctrl+C` 会清理引擎子进程。

## 不加载权重的检查

```bash
bash scripts/run/serve_vllm.sh --check-only
```

该命令读取版本、结构注册和本地配置/tokenizer，可查询驱动信息；不加载权重、不初始化 CUDA、不执行 GPU kernel。它会读取 `.env.local` 并启用与实际引擎一致的兼容层。检查通过不等于 GPU 推理已验证。

集群中需检查计算节点时提交：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_vllm.sh --check-only
```

也有较小资源的 `scripts/slurm/check_vllm.sh`（1 GPU、4 CPU、16 GB），直接调用诊断器，不默认启用引擎兼容层，可能报告旧 tokenizer 属性警告。

## 独立引擎与已有服务

通常无需独立启动。需要单独管理引擎时：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_vllm.sh --model-dir ckpts/Hy-MT2-1.8B
# 普通独立 GPU 机器：
# bash scripts/run/serve_vllm.sh --allow-local --model-dir ckpts/Hy-MT2-1.8B
```

此服务没有网页，持续前台运行。正式 Demo 连接已准备的服务时使用：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh --no-manage-vllm --mt-api-base http://VLLM_HOST:8000/v1 --mt-served-model hy-mt2 --mt-model-dir ckpts/Hy-MT2-1.8B
```

`VLLM_HOST` 为 Demo 计算节点可达的服务地址。不同 Slurm 作业不保证同节点，`127.0.0.1` 只代表本节点；外部服务由其启动者清理，不随 Demo 自动退出。外部服务精度由服务端决定，客户端不能改变已经加载的精度。

`REALTIME_VLLM_PYTHON` 可覆盖引擎 Python，留空使用 Demo 的 Python；已有单环境无需配置它。独立引擎端口读 `REALTIME_VLLM_PORT`，托管引擎端口从 `REALTIME_VLLM_API_BASE` 读取。客户端默认不继承 HTTP 代理，本机请求不会误走公网代理。

## 资源与性能配置

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `REALTIME_VLLM_GPU_MEMORY_UTILIZATION` | `0.4` | 按整卡显存制定引擎预算 |
| `REALTIME_VLLM_MAX_MODEL_LEN` | `8192` | 服务端总上下文上限 |
| `REALTIME_VLLM_MAX_NUM_SEQS` | `1` | 引擎并行序列数 |
| `REALTIME_VLLM_MAX_BATCHED_TOKENS` | `512` | 分块预填充 token 预算 |
| `REALTIME_VLLM_STARTUP_TIMEOUT` | `1800` | 托管器等待健康检查秒数 |
| `REALTIME_VLLM_REQUEST_TIMEOUT` | `180` | 翻译 HTTP 请求超时秒数 |
| `REALTIME_VLLM_ENFORCE_EAGER` | `0` | 设为 1 诊断编译/图捕获问题，可能变慢 |
| `REALTIME_VLLM_TOKENIZER_COMPAT` | `1` | 旧引擎 tokenizer 兼容层 |

默认开启前缀缓存与分块预填充。80 GB 卡上的 0.4 是约 32 GB 引擎预算起点，不是权重大小、算力占比或整个应用峰值；还需留出 ASR、MMS、Seamless 及临时张量空间。7B 需重新评估预算。

输入默认最多 4096 token（含提示词/模板），输出最多 1024。客户端上限和服务上下文都需容纳实际输入输出。达到上限或异常断流时明确报错，不把残缺译文标为完成。

前缀缓存节省重复输入的预填充工作，不自动复用上一轮完整译文。长原文仍会增加生成耗时。跨进程共用 GPU 时，ASR 的进程内高优先级流不保证优先于 vLLM，应同时观察原文与译文的延迟。对照方法见[实时页面验证](realtime_testing.md)及 [Hy-MT2 对照](hy_mt_testing.md)。

需要回退时，正式脚本增加 `--mt-backend hy-mt2` 使用原生 Transformers，或 `--mt-backend seamless` 使用 Seamless 文本模型；只保留文件端到端翻译使用 `--no-realtime --file-translation speech`。
