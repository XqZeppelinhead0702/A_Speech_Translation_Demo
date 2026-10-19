# 正式 Demo：两页运行与验证

[返回 README](../README.md) · [环境与模型](environment.md) · [Slurm 部署](deploy_slurm.md) · [vLLM 说明](vllm.md)

## 两个页面分别做什么

| 页面 | 默认处理流程 | 输入语言 | 结果显示 |
| --- | --- | --- | --- |
| 语音翻译 | SeamlessM4T 端到端：音频直接生成译文，另行转写用于显示原文 | 默认 MMS 自动检测，也可手动指定 | 先显示原文，再显示译文 |
| 同声传译 | Nemotron 连续流式 ASR → Hy-MT2/vLLM 文本翻译 | 手动指定 | 原文独立更新，译文跟进；停止后等待最终结果 |

文件页默认选择 **SeamlessM4T（端到端）**，翻译直接消费音频特征，不把识别原文交给文本翻译模型。可选级联使用“Nemotron 非流式识别 → Hy-MT2/vLLM”，整段音频一次预处理，识别过程中显示原文进度。它与实时页共享 Nemotron 和文本翻译器，不重复加载权重。粤语等当前 Nemotron 映射未覆盖的文件输入仍由 SeamlessM4T 转写，并在日志标明兼容路径。文件页的自动语言检测不会应用到实时页。

“非实时/实时”描述音频的输入与输出时机，“端到端/级联”描述模型组合，非实时也可以采用级联。Nemotron 的[官方模型卡](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#-transformers-usage)提供 Offline transcription，当前 Transformers processor 也支持 `is_streaming=False`，可用于完整文件识别。文件级联现在使用这一离线 API，默认端到端选项仍用 SeamlessM4T。

“语音翻译”和“同声传译”在独立切换栏中直接显示。文件页内部通过“翻译模型”下拉框选择端到端或级联，采用与整体主题一致的配色。

### 演示播放与录音自动结束

文件页点击“识别并翻译”（包括 Ctrl/⌘ + Enter）后，会从头播放当前音频，同时开始处理。按钮下方出现独立试听播放器，可暂停、拖动进度或停止试听；停止试听不会取消翻译。播放器独立于上传控件，处理时锁定输入不会打断声音。清空、换示例、重新录音或切换页面时会停止试听。浏览器禁止自动播放时，可手动点击试听播放键，识别与翻译仍会继续。此声音用于演示文件内容，不代表文件识别在按照播放进度流式处理。

实时麦克风默认启用“讲话后静音自动结束录音”：检测到讲话后，连续静音 **2.5 秒**便停止采集，确认尾部音频上传完成后，继续等待最终识别和翻译结果。开始录音后的安静等待不会直接触发结束；短暂停顿也不会切断会话。静音等待可选 1.5、2.5 或 4 秒；长停顿较多时可选择 4 秒或关闭自动结束，改用“停止并等待最终结果”。设置在每次录音开始前生效。

这项检测在浏览器中根据音频能量和噪声基线判断，不增加模型、GPU 占用或网络请求，也不会把原文或译文拆成固定片段。它判断的是声学停顿，不保证句意已经结束；轻声、远距离录音或持续背景噪声可能影响判断，可手动停止或关闭自动结束。检测只用于麦克风，示例音频始终完整回放。

两个页面共用一个网页端口与公网 URL。现有“浏览器 → Gradio 分享/登录节点 → SSH → 计算节点”通路保持原样；vLLM 只在计算节点的 `127.0.0.1:8000` 提供服务，原文和译文先在计算节点内部传递，再由 Demo 发给浏览器。浏览器不直接访问 vLLM，也不需要转发 8000 端口。

## 已有环境如何运行

已用 `st_demo` 跑通过 vLLM 实时测试时，直接沿用这个环境，不需要另建环境或重复安装。准备以下完整本地模型目录：

```text
ckpts/
├── seamless-m4t-v2-large/
├── mms-lid-256/
├── nemotron-3.5-asr-streaming-0.6b/
└── Hy-MT2-1.8B/
```

模型路径可通过 `.env.local` 覆盖。原有 Python、FFmpeg、SSH、登录节点及代理配置继续使用；没有新增配置时，正式入口默认相当于：

```bash
DEMO_REALTIME=1
DEMO_FILE_TRANSLATION=speech
DEMO_MT_BACKEND=hy-mt2-vllm
DEMO_MANAGE_VLLM=1
```

如果已有 `.env.local` 显式设置 `DEMO_FILE_TRANSLATION=text`，它会覆盖新默认值；改为 `speech`，或提交时追加 `--file-translation speech`。未配置此项时无需修改。

正式入口使用 `DEMO_MT_BACKEND`，不读取旧测试入口的 `REALTIME_MT_BACKEND`。模型、精度与性能设置使用 `REALTIME_ASR_*`、`REALTIME_HY_MT_*` 和 `REALTIME_VLLM_*`，完整清单见[配置说明](configuration.md)。

在项目根目录提交原来的正式入口，例如登录节点中转：

```bash
mkdir -p outs
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh
```

将 `GPU_PARTITION` 替换为实际分区，也可按集群规则指定 A100 80 GB 类型。默认申请 **1 GPU、8 CPU、64 GB 主机内存**。不要同时保留占用同卡/端口的旧测试作业。计算节点直连公网用 `serve_direct.sh`，固定域名用 `serve_public_tunnel.sh`，具体配置仍见 [Slurm 部署](deploy_slurm.md)。

启动顺序为：原网络预检查 → 启动 vLLM 并等待健康检查 → 加载 Nemotron、语言检测和 SeamlessM4T → 启动正式页面 → 打印公网 URL。首次引擎编译可能较慢，默认最多等待 1800 秒，不受网页分享的 90 秒启动超时限制。登录节点等待后端就绪的窗口相应延长为 2400 秒，避免旧的 300 秒限制提前终止作业；可设 `DEMO_RELAY_BACKEND_TIMEOUT`，应大于引擎等待时间加其他模型加载耗时。两个页面都准备好后才提供页面服务。

日志位置：

- `outs/demo-作业号.out/.err`：正式应用与中转日志，查找 `PUBLIC_URL`。
- `outs/vllm-engine-作业号.log`：vLLM 引擎和工作进程日志。

用 `scancel 作业号` 停止；托管引擎会随 Demo 一起清理。引擎异常退出时，Demo 也会结束，避免页面长期显示可用却无法翻译。`--no-manage-vllm` 连接的外部引擎由其启动者管理，不会被 Demo 停止。

旧的本地 `serve_demo.sh` 如果最终仍调用 `app.py`，也会获得新默认功能，但它自己的 CPU/内存资源头不会自动更新；建议改用上述通用入口，或提交时明确 `--cpus-per-task=8 --mem=64G`。

## 建议的首次验证顺序

1. 打开公网 URL，在“语音翻译”页点击一个示例，确认语言和音频，提交后应自动播放输入音频，先见原文，再见译文。
2. 确认默认“SeamlessM4T（端到端）”完成翻译，再选择级联对照同一样本。用“自动检测”检查语言识别。
3. 在“同声传译”页选择一个示例，点击“播放并实时测试示例”。声音持续播放；原文不等待翻译，译文保留已有字幕并跟进，不从开头重新播放文字。结束后等待“处理完成”。
4. 手动指定说话语言，允许麦克风访问，开始录音，讲话后安静等待 2.5 秒，应自动停止并补齐最终原文与译文。再关闭自动结束，测试手动停止，确认两种方式均释放会话资源。
5. 译文延迟明显时展开“延迟诊断”，比较首条 ASR、首条译文、原文交付耗时和网络 RTT；引擎耗时则查看 vLLM 日志。

两个页面之间可随时切换，但一个实例同时只允许一个模型任务。实时会话尚未结束时提交文件会收到提示；文件仍在处理时开始实时会话也会收到提示，不会自动中断当前任务。录音停止后可能仍在补齐译文，需等待处理完成或取消。

实时字幕沿用已验证的累计原文/译文设计，仍可能修订已显示内容；还没有增加按语义块永久锁定前文的机制。长会话、多人同时讲话及噪声环境需另外评估，vLLM 加速不保证固定的端到端时延。

## 切换模型或回退

以下参数都可直接追加在正式脚本之后，脚本会传给 `app.py`。

使用 7B，其他流程相同，但需重新评估显存预算：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh \
  --mt-model-dir ckpts/Hy-MT2-7B
```

改用 Transformers 加载 Hy-MT2，不启动 vLLM：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh \
  --mt-backend hy-mt2 --mt-model-dir ckpts/Hy-MT2-1.8B
```

仅保留原文件/录音直接语音翻译；不加载 Nemotron、Hy-MT2，也不启动 vLLM：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh \
  --no-realtime --file-translation speech
```

只关闭实时页、文件页改用 vLLM 级联，使用 `--no-realtime --file-translation text`；此时仍加载文件识别需要的 Nemotron。保留实时页并将文件页默认选项改为级联，使用 `--file-translation text`；不传该参数时文件页默认端到端。

连接已经准备好的 vLLM 服务：

```bash
sbatch --partition=GPU_PARTITION scripts/slurm/serve_login_relay.sh \
  --no-manage-vllm --mt-api-base http://VLLM_HOST:8000/v1 \
  --mt-served-model hy-mt2 --mt-model-dir ckpts/Hy-MT2-1.8B
```

`VLLM_HOST` 替换为计算节点可达的服务地址，两端 tokenizer/模型规格必须一致。不同 Slurm 作业未必在同一节点，不能直接假定另一作业的服务能经 `127.0.0.1` 访问。默认托管方式更适合单卡测试。

## 普通 GPU 机器与 CPU 预览

普通独立 GPU 机器沿用公开脚本，自动托管引擎也支持该方式：

```bash
bash scripts/run/serve_share.sh
# 已配置 HTTPS 反向代理时：
# bash scripts/run/serve_local.sh
```

集群环境仍应提交 Slurm。独立启动引擎的脚本在非 Slurm 机器需要显式 `--allow-local`；正式应用托管时会自动传入，无需手动执行第二个脚本。

不加载 GPU 模型，仅检查界面：

```bash
bash scripts/run/serve_local.sh --demo-mode
```

文件页对新展示样本仅预览音频；实时页明确显示 **MOCK** 占位字幕，只验证录音、回放与消息流程，不能用于评价翻译质量或模型速度。

## 资源说明

新增的常驻模型包括 Nemotron、MMS、SeamlessM4T 和单独进程中的 Hy-MT2。默认 vLLM `gpu_memory_utilization=0.4` 是整卡预算，80 GB 卡上约 32 GB；它不代表只用 40% 算力，也不等于权重大小。除此之外还需留出其他三个模型、运行缓存和临时张量的空间。

当前没有提供四个模型总峰值的实测数据；部署时优先用 A100 80 GB 进行整体验证。若显存不足，先查看引擎日志与模型加载阶段，再调整 `REALTIME_VLLM_GPU_MEMORY_UTILIZATION`、模型规格或语言检测后端，不要通过申请更多主机内存来解决 GPU 显存不足。

文件识别排错：日志会记录 `File ASR started/completed`、后端名、音频时长、转写耗时和 `File translation started`，不写入完整原文。`DEMO_FILE_ASR_TIMEOUT` 默认 120 秒，Nemotron 在生成步骤/输出回调处检查超时与取消；GPU 底层调用无法由 Python 强行抢占。转写异常不会继续翻译或把空文本标记为完成。
