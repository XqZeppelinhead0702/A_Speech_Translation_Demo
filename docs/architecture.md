# 架构与处理流程

[返回 README](../README.md) · [运行与验证](formal_demo.md) · [实时接口](realtime_development.md)

## 三条处理路径

| 页面 / 选项 | 识别原文 | 生成译文 | 输入语言 |
| --- | --- | --- | --- |
| 语音翻译：SeamlessM4T（默认） | SeamlessM4T 转写整段音频 | SeamlessM4T 直接消费音频 | MMS 自动检测或手动指定 |
| 语音翻译：Hy-MT2 级联（可选） | Nemotron 非流式识别；未覆盖语言使用 SeamlessM4T 转写 | 将最终原文交给 Hy-MT2 | 同上 |
| 同声传译 | Nemotron 连续流式识别 | Hy-MT2 随原文版本跟进 | 手动指定，限当前 Nemotron 映射支持的语言 |

文件页默认的译文不经过“识别文本 → 文本翻译”。级联选项与实时页共用文本翻译器，默认由 vLLM 执行，也可启动时改为原生 Transformers Hy-MT2 或 Seamless 文本翻译。两页均输出文字，不生成译后语音。

## 模型和进程

```text
浏览器：Gradio 两页 + AudioWorklet 采集/回放 + 字幕更新
    │ 一个 HTTPS URL；实时优先 WebSocket，失败回退 HTTP
    ▼
Demo 进程（默认 7860）
    ├── SeamlessM4T：文件端到端翻译 / 兼容转写
    ├── MMS：文件语言检测，可切换 SpeechBrain CPU
    ├── Nemotron：文件离线识别 / 实时流式识别，复用一份权重
    └── Hy-MT2 客户端：tokenizer + 本机流式 HTTP
                           │
                           ▼
vLLM 进程（默认 127.0.0.1:8000）：一份 Hy-MT2 权重
```

默认四个模型在同一 GPU 作业内运行；vLLM 和 Demo 使用同一 Python 环境。自动托管器负责启动、等待健康检查和退出清理。`--no-manage-vllm` 改为连接已有服务。原生 Hy-MT2 在 Demo 进程中加载，不启动 vLLM。

`InferenceCoordinator` 让整个文件任务与实时会话互斥；实时会话内部的识别和翻译独立工作，原文无需等待翻译完成。当前实例同时支持一个模型任务，不是多用户并行推理服务。

## 实时输入与字幕

1. 浏览器每约 160 ms 交付采样，上传队列分配序号并检查确认。网络小包不是句子边界，不会重置 ASR 缓存。
2. Nemotron 每次产生新原文便发布递增版本；翻译任务保存对应原文快照和版本。尚未开始的普通草稿允许合并，自然停顿、标点和最终检查点保留处理。
3. 原文优先独立推送，译文通过流式候选和 `SubtitleProjection` 更新。页面保留已有字幕，追加新文字或修订尾部，避免每轮从头逐字重播。
4. 麦克风默认在讲话后静音 2.5 秒触发结束，可调整或关闭。检测在浏览器进行，不需要新增 VAD 模型；示例回放不启用它。
5. 停止时先补发 AudioWorklet 尾部采样，等待所有上传确认，再发送结束信号。页面继续接收结果，直到最终识别和翻译同步完成。

当前仍会对累计原文重新翻译，尚未永久提交语义块，也没有跨翻译请求复用完整 decoder 状态。显示稳定和 vLLM 加速不会消除长输入的全部计算量；上下文过长会明确报错，不静默截断。

## 模块分工

| 位置 | 职责 |
| --- | --- |
| `app.py`、`language_detection.py` | 正式界面、文件任务、模型初始化和语言检测 |
| `realtime/backend.py`、`stream_audio.py`、`audio.py` | 模型适配、连续重采样与特征、PCM 校验 |
| `realtime/service.py`、`subtitles.py` | 会话生命周期、ASR/MT 版本调度、字幕投影 |
| `realtime/hy_mt.py`、`vllm_mt.py`、`translation.py` | 文本翻译接口、后端选择与流式输出 |
| `realtime/api.py`、`socket.py` | 同域 HTTP / WebSocket、会话归属与认证 |
| `realtime/integration.py`、`engine.py`、`cuda_execution.py` | 两页互斥、引擎管理、进程内 CUDA 调度 |
| `ui/realtime.*`、`audio_transport.js`、`endpoint_detector.js` | 实时界面、上传/确认、麦克风停顿结束 |
| `ui/app.js`、`file_playback.js`、`style.css` | 文件页主题、状态显示与提交时试听 |
| `sample_catalog.py`、`data/test_samples/realtime_samples.json` | 两页共用的 10 条展示音频 |
| `scripts/run/`、`scripts/slurm/` | 环境初始化、网络入口与正式运行脚本 |

协作者接入时优先复用公开 API 和版本协议，避免重新实现模型加载、终止收尾和字幕排序。字段定义见[实时开发说明](realtime_development.md)。
