# Hy-MT2 模型选择与效果对照

[返回 README](../README.md) · [环境准备](environment.md) · [vLLM](vllm.md) · [实时验证](realtime_testing.md)

## 1.8B 与 7B

当前适配原始 dense 权重 Hy-MT2-1.8B / 7B。两者使用相同接口、参数和字幕协议，切换只需更换完整模型目录并重启。7B 需要更多权重显存，实际吞吐和效果应使用同一样本实测，不能假定没有额外时延。

保留各自 tokenizer、配置及 generation config，不能混用 EOS/tokenizer；vLLM 服务与 Demo 客户端使用同一套模型文件。当前未接入 FP8、GGUF 或 MoE 权重。

## 默认 vLLM 运行

以登录节点中转为例，在项目根目录提交：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh --mt-backend hy-mt2-vllm --mt-model-dir ckpts/Hy-MT2-1.8B
```

文件页的 SeamlessM4T 端到端仍是默认选项；选择文件级联或进入同声传译才使用 Hy-MT2，两处共用引擎。换 7B 时改为：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh --mt-model-dir ckpts/Hy-MT2-7B
```

先结束旧作业，避免端口/GPU 资源冲突。按实际显存调整 `REALTIME_VLLM_GPU_MEMORY_UTILIZATION`，默认 0.4 是整卡预算，不是权重大小。其他网络环境替换为对应正式入口，应用参数不变。

## 原生 Transformers 对照

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh --mt-backend hy-mt2 --mt-model-dir ckpts/Hy-MT2-1.8B
```

该方式不启动 vLLM，在 Demo 进程加载 Hy-MT2。已有环境无需修改，新环境依赖见[环境说明](environment.md#环境基线与依赖选择)。同一份原始权重可在两种后端使用，CUDA/tokenizer 兼容性仍需分别验证。

## 容量与参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `REALTIME_HY_MT_PRECISION` | `bfloat16` | 独立于 Seamless/MMS/ASR |
| `REALTIME_HY_MT_MAX_SOURCE_TOKENS` | `4096` | 输入上限，含提示词和模板 |
| `REALTIME_HY_MT_MAX_NEW_TOKENS` | `1024` | 单次输出上限 |
| `REALTIME_HY_MT_SAMPLING` | `0` | 默认贪心；1 开启采样 |
| `REALTIME_MT_STREAMING` | `1` | 发布生成中候选 |

对应 CLI 为 `--mt-precision`、`--mt-max-source-tokens`、`--mt-max-new-tokens`、`--mt-sampling`。vLLM 的 `REALTIME_VLLM_MAX_MODEL_LEN` 也需容纳输入与输出，增加客户端上限不会自动扩大服务上限。超限明确报错，不静默截断。

## 如何比较

固定样本、语言方向、网络入口、采样设置和播放节奏，分别运行 1.8B/7B 或原生/vLLM。用文件页级联比较最终译文，用实时页比较首条原文、首个译文片段、完整译文及收尾等待。首轮可能含预热，应重复测试。

主字幕保留已有内容并追加或修订尾部，完整结果仍可能改写前文。累计源文本越长，重新生成的工作量越大；流式候选与前缀缓存不等于已实现增量 MT。详见[延迟诊断](realtime_testing.md#延迟诊断)。

官方资料：[1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[7B](https://huggingface.co/tencent/Hy-MT2-7B)。
