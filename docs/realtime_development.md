# 流式识别、级联翻译与前端同步协议

[返回 README](../README.md) · [测试步骤](realtime_testing.md)

实时后端使用 Nemotron 3.5 cache-aware RNNT，持续消费音频特征生成器；原文驱动 SeamlessM4T v2 或 Hy-MT2 文本翻译。正式“同声传译”Tab 已复用这一实现，默认 Hy-MT2/vLLM；文件页默认 SeamlessM4T 端到端语音翻译，可选级联才共享同一文本翻译器。实际运行见[正式 Demo](formal_demo.md)。

## 模块

```text
realtime/
├── __init__.py       # RealtimeService / create_service
├── service.py        # 音频接收、ASR 工作线程、独立翻译线程、版本与确认队列
├── api.py / socket.py # 正式同域 HTTP / WebSocket 与会话归属校验
├── integration.py    # 文件任务与实时会话互斥
├── engine.py         # 正式应用托管 vLLM 的启动、健康检查与清理
├── backend.py        # Nemotron 流式适配、Seamless T2T、CPU mock
├── hy_mt.py          # Hy-MT2-1.8B/7B 原生文本翻译适配
├── vllm_mt.py        # 独立 vLLM 服务的流式 HTTP 适配，无本进程翻译权重
├── translation.py    # 翻译后端选择、共享配置和测试 CLI 参数
├── subtitles.py      # 候选译文与可读字幕的独立显示状态
├── cuda_execution.py # ASR/MT 独立计算流，ASR 优先调度
├── stream_audio.py   # SoXR 连续重采样、STFT 窗口、自然停顿统计
└── audio.py          # 输入 PCM 校验与归一化
```

factory 只在显式调用时加载模型。直接调用默认要求 Slurm；正式入口显式传入 `require_slurm=False`，以同时支持 Slurm 和普通独立 GPU 机器。集群仍通过 Slurm 提交。`create_service(pipeline="asr")` 不创建翻译器；`pipeline="cascade"` 使用所选文本后端。`mock=True` 不加载权重。手动指定源语言，拒绝 `auto`；源语言代码映射在 `ASR_LANGUAGES`，目标语言沿用正式 Demo。

需要正式应用共用已加载的完整 SeamlessM4T 时，可传 `create_service(translator=existing_translator)`。T2T 使用其推理锁，ASR 有独立锁；不能在 ASR 等待新音频期间持有翻译锁。独立测试默认使用精简文本模型，与原 Demo 不同进程。正式入口已通过 InferenceCoordinator 将文件任务与整个实时会话互斥；自定义应用复用模型时也需保留这种调度，避免长文件影响实时会话。

选择 Hy-MT2 时，可调用 `create_service(pipeline="cascade", mt_options={"backend": "hy-mt2", "model_dir": "ckpts/Hy-MT2-1.8B"})`，7B 只需换目录。它独立加载所选翻译模型，不能复用 SeamlessM4T 权重。两种适配器都返回完整译文字符串，原音频接口、事件与版本协议相同，前端无需增加模型专用的字幕协议。后端在创建会话前验证翻译语言支持；ASR-only 不创建翻译适配器。测试见 [Hy-MT2 测试](hy_mt_testing.md)。

### 可选 vLLM 服务

`create_service(pipeline="cascade", mt_options={"backend": "hy-mt2-vllm", "model_dir": "ckpts/Hy-MT2-1.8B", "api_base": "http://127.0.0.1:8000/v1", "served_model": "hy-mt2"})` 使用独立进程中的翻译模型。服务须提前启动；本进程只加载 tokenizer 并检查 API，不启动或安装 vLLM。

该适配器沿用 `translate()` / `translate_stream(on_text=..., stop=...)`，使用相同原文版本和字幕投影，不修改前端协议。只有正常收到 `finish_reason=stop` 和流结束标记才返回完整译文；截断/断连报错并由服务回退到上一份完整字幕。取消时关闭 HTTP 流，允许 vLLM 中止对应请求。版本兼容、同卡资源预算与测试见 [vLLM 部署](vllm.md)。

## 音频通路

浏览器 AudioWorklet 每约 160 ms 交付浮点采样，上传层编码为 PCM16。正式页优先使用二进制 WebSocket 流水传输，音频包不逐块等待回复；HTTP 回退将已经等待的连续采样合并为最多 2 秒的包，避免公网 RTT 造成串行积压。包序号在发送时分配，小块传输和合包都不是语义切句。服务端转单声道，以保持过滤状态的 SoXR 连续转换到 16 kHz。STFT 窗口按全信号位置对齐，FFT 重叠只用于特征计算；编码器、RNNT 解码器缓存不随块重置。

ASR 每会话只调用一次 `generate(input_features=generator, streamer=receiver, ...)`，等待音频到来，直至输入 EOF。显式限定 `max_new_tokens=16384`，避免实验接口默认极大的生成长度；提前结束或达到上限明确报错。四档 lookahead 对应所需帧数严格校验，最后不足一块的特征补齐。当前 processor 会忽略 `is_last_audio_chunk`，后续尾块直接调用 feature extractor，将尾部补齐放在 preemphasis 之后；CPU 检查对比完整音频的有效特征，避免边界漏样。

自定义 token receiver 按 token 解码累计文本，不依赖空格才发布，因此中文也能持续更新。模型真实速度、精度与 CUDA 兼容性仍需 GPU 作业验证。[官方流式接口](https://huggingface.co/docs/transformers/model_doc/nemotron3_5_asr)

## Python 接口

```python
service = create_service(mock=True, pipeline="cascade")
session = service.start_session(source_lang="cmn", target_lang="eng", owner_id="browser-owner")
sid = session["session_id"]
service.push_audio(sid, chunk_seq=1, sample_rate=48000, audio=pcm_array, owner_id="browser-owner")
result = service.poll_results(sid, cursor=0, owner_id="browser-owner")
service.finish_session(sid, last_chunk_seq=1, owner_id="browser-owner")
# Continue poll_results until finished/error/cancelled.
service.close()
```

音频序号从 1 开始，重试已确认块不会再次处理，缺块拒绝接收；每会话采样率固定。音频为整数 PCM 或归一化 float 数组，可带声道维。`push_audio` 和 `poll_results` 不等待模型推理。所有操作验证 owner；正式前端在服务端生成归属标识，不让全体用户共用。

`cancel_session(sid, owner_id=...)` 唤醒等待音频的 ASR，删除未开始任务，禁止旧结果发布。GPU 已执行调用不能立即打断。模型 mutable streaming 状态只允许一个活动会话；取消之后若 ASR 还没退出，新会话会提示稍后再试。

## 事件与快照

ASR 每次不同文本产生递增 `asr_version`；`event_seq` 对两类事件统一递增。例子：

```json
{"type":"asr","event_seq":1,"asr_version":1,"transcript":"我想订票","is_final":false}
{"type":"asr","event_seq":2,"asr_version":2,"transcript":"我想订明天去上海的票","is_final":false}
{"type":"translation","event_seq":3,"asr_version":2,"translation_version":1,"translation_source":"我想订票","translation":"I want to book a ticket.","is_final":false}
```

真实事件还包含 `session_id`、语言、`observed_seconds` 和 `server_time_ms`；翻译事件附带 `reason`、排队及推理耗时。`observed_seconds` 是单调时钟的会话耗时，`server_time_ms` 是 UTC Unix 毫秒，用于浏览器校时后的显示延迟估计，不能直接与未经校时的另一设备时间相减。`asr_version` 表示发布译文时的当前 ASR 版本，**`translation_version` 才是译文真正对应的原文版本**。不要把两者当成同一个字段。

`poll_results` 返回游标后的 `events`、`cursor`、`status`、`error` 以及当前 `snapshot`：

```json
{
  "transcript":"我想订明天去上海的票",
  "asr_version":2,
  "asr_final":false,
  "translation":"I want to book a ticket.",
  "translation_version":1,
  "translation_source":"我想订票",
  "translation_final":false,
  "lag_versions":1,
  "translation_pending":true
}
```

快照还包含首条原文、译文等待时间、首块音频到达时刻 `first_audio_seconds` 及合并的草稿数。三者都从会话创建计时，用 `first_asr_seconds - first_audio_seconds` 可排除建连等前期等待，但仍包含声音上下文积累与前处理，不能当作纯推理耗时。外层包含音频已接收、已处理、音频缓冲时长、确认任务数、pipeline、backend 及响应的 `server_time_ms`。ASR-only 的译文始终为空，页面明确标记没有翻译模型。

## 避免漏显示和错误覆盖

1. ASR 更新与翻译完成各自发布事件，原文不等译文，译文也不因原文更新被清空。
2. 每个翻译任务保存原文快照、版本，完成后发布，即使 ASR 此时已经多次更新。
3. 待翻译的普通草稿合并为最新版本；完整 ASR 更新事件仍保留。标点、自然停顿和停止时的确认版本入独立有界队列，不参与合并。
4. 页面主区域显示 `snapshot`；旧版译文可以在等待期间保留，但必须标注对应版本。只有完成版本大于等于已显示译文版本时才更新，不让旧结果覆盖新结果。
5. 历史按事件游标消费一次。可以批量拿到多条 ASR 更新，即时主视图展示最新累计文本，记录保留中间快照；不承诺每个瞬时 token 都单独翻译或逐帧显示。
6. 停止先 flush 浏览器尾部并等待上传确认，再结束输入。`draining` 等 ASR EOF、保留的确认任务和最终译文全部完成，最终版本一致时才进入 `finished`。

当前文本翻译以本会话累计原文为上下文，可以随新信息改写之前的译文；不是专用增量 MT，也不复用旧源文本的 decoder KV。第一版不冻结译文前缀，避免语序变化导致错误确认；确认任务是必须处理的检查点，不是永远不能改写的文本。

## Hy-MT2 生成中译文

`translate(text, source, target)` 仍返回完整译文字符串。新增 `translate_stream(text, source, target, on_text=callback, stop=event)` 同样返回经过 EOS 校验的完整译文，但生成期间可多次调用 callback，传入当前累计的候选译文。回调在翻译工作线程执行，不持有会话锁等待 GPU。中文按解码文字发布，过滤尚未组成完整字符的 UTF-8 token，约每 80 ms 合并更新，最后强制刷新草稿。取消后在下一次 token 回调处中止生成；GPU 已经执行的运算仍不能立即抢占。

原 `translation` 事件、`translation_version` 与 `translation_source` 保持完整结果的含义。快照新增：

- `translation_draft`、`translation_draft_version`、`translation_draft_source`：生成中的文字及其源版本；可能对应较早的 ASR，但不会冒充最新原文或最终结果。
- `translation_draft_seq`：本会话单调递增的草稿变化序号，生成中更新和清除都递增；它独立于历史 `event_seq/cursor`。
- `translation_draft_server_time_ms`：最近一次草稿生成时间。
- `first_translation_draft_seconds`：从会话创建到首个非空可读草稿；原 `first_translation_seconds` 仍是首份完整译文完成时刻。

草稿不逐 token 加入历史列表，避免长会话超出更新记录容量。WebSocket 必须同时比较草稿序号与历史游标，不能仅在完整事件变化时推送；HTTP 响应包含同一快照。命令行回放额外记录 `type=translation_draft`，这些记录没有历史事件序号，不能混入原事件游标。

草稿不参与完成判定，也不能作为重复检查点的缓存。EOS 缺失、空输出或生成异常时明确失败并清除草稿，保留最近完整结果；取消后拒绝后续回调。只有原 ASR/MT 最终版本一致且所有必需任务完成，才报告 `finished`。设置 `REALTIME_MT_STREAMING=0`，或直接 `create_service(stream_translation=False, ...)` 可关闭；不支持回调的翻译器自动沿用完整结果路径。旧前端可继续只读取完整结果字段，新前端读取草稿即可增加实时显示。

这一输出通路不实现永久前缀提交或跨次推理 KV 复用。累计源文本变化时，原 decoder cache 不能直接复用；提前显示输出改善等待观感，但没有保证长期的完整译文延迟保持不变。[Transformers streamer 接口](https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.TextStreamer)

## 字幕显示状态：避免每次从头回放

一次 `generate` 的候选长度从零增长，不能直接替换上一轮已经显示的长字幕。`SubtitleProjection` 为每个会话保存当前显示和最近经过 EOS 校验的完整结果，独立于原始 `translation_draft`。

主视图读取以下新快照字段：

- `translation_display`、`translation_display_version`、`translation_display_source`：实际应显示的字幕及其原文版本。
- `translation_display_provisional`：是否包含本轮尚未经过 EOS 校验的文字；false 只表示对应生成已完整，不表示永远不能修改。
- `translation_display_seq`、`translation_display_server_time_ms`：实际字幕状态变化的序号和生成时间，用于显示更新诊断。

新候选短于当前字幕时不替换，完全重复时也不推进显示版本。候选沿当前字幕继续增长时追加；候选已经覆盖当前长度且只改写较短尾部时允许局部修订（最多最后 24 个字符且约占已有文字末尾的五分之一，短字幕至少预留 2 个字符）。这是显示上的保守边界，不是切句或提交规则。更大改写等待完整结果后一次更新，既不反复从头逐字播放，也不永久保留翻错的前文。首次生成仍可逐步显示，正常生成轮次切换不清空主视图。

已有 `translation` 与 `translation_draft` 字段仍保留。客户端可继续消费原完整结果，新的字幕前端应使用 `translation_display`；原始 draft 仅用于调试模型生成过程，不能将它直接接回正常字幕区域。主视图的“对应原文”应使用 display 的 source/version，不能拿最新 ASR 的版本替换它。命令行回放用 `type=subtitle` 单独记录实际可读字幕变化，原 `type=translation_draft` 仍只记录原始候选；两者都不混入历史事件游标。

取消或失败时回滚到最近完整生成；正常完成清除原始草稿时保留可读字幕。新候选被隐藏期间不会假装更新了字幕时间戳，也不把短候选和旧结果尾部拼成一个混合版本。文本节点沿共同前缀追加或修改变化范围，没有从头重播的动画。

这不是语义块永久锁定，也不是跨次推理缓存优化：生成耗时和字幕可读性需要分开评估。源语言没有变化的旧前缀仍可能因为新上下文或语序而需要纠正，完整结果保留该能力。

## 语义块显示扩展（设计，尚未实现）

适合持续阅读的界面通常需要两部分：**已提交的原文/译文块保持稳定，当前未完成块允许修订**。当前版本仍显示整场会话的文本快照，并未实现块提交。Nemotron 适配器不断追加 token 再解码累计原文，通常以增长为主，少量拼接或标点变化不代表重新识别整个历史；测试页按共同前缀更新文本节点，新增部分直接追加。T2T 则确实反复翻译累计原文，新增信息可以改变较早的译文，当前缺少“这一块已结束且不可改写”的约束。

期望的显示可以是：

```text
已提交块 1：明天上午十点开会。
已提交译文：The meeting is at ten tomorrow morning.

当前草稿块 2：请大家提前…… → 请大家提前准备材料。
当前草稿译文：Please prepare… → Please prepare the materials in advance.
```

这是一项文本状态管理扩展，可以继续使用现有 ASR/T2T，不要求换模型或增加另一个大型模型。音频持续输入，ASR 编码器和解码器缓存保持；语义块用于文本提交和翻译调度。第一版可按以下规则实现，再用真实音频调节：

1. **识别稳定性与语义边界分别判断。** ASR 原文稳定，不代表句子已经说完。结合稳定文本前缀、句末标点、自然停顿和后续少量上下文选择边界；不能只因过了固定秒数或译文两次相同就提交。中文、英文应分别处理字符/词边界。没有标点时先用停顿辅助，但停顿也可能出现在句中，不保证语义完整。
2. **维护块标识和状态。** 每个块有 `block_id`、源文本版本、原文草稿/已提交状态、译文草稿/待最终翻译/已提交状态。ASR 提交块 1 后可以立刻显示块 2 的草稿；块 1 译文未完成时显示“翻译中”，两侧不必同时提交。只在块 1 最终源文本的翻译完成后锁定其译文，避免冻结较早草稿。
3. **翻译调度按块隔离。** 只合并同一未提交块的待执行草稿。已经提交的每个源块都保留最终翻译任务；当前块更新不会删除上一块的最终任务。返回结果用 `block_id + source_revision` 定位，旧结果不覆盖新草稿，锁定块不随新块改变。停止输入时提交最后一个块，再等所有已提交块的译文完成。
4. **控制翻译上下文。** 第一版可翻译完整的当前语义块，避免每次重新生成全部历史；这样上下文变短，但跨句指代可能变差。SeamlessM4T T2T 没有“把历史作为只读提示、只输出当前块”的专用接口，不能简单加聊天式提示语或把整段新译文按字符硬切。需要上下文时，应设计并验证重叠窗口与输出对齐，或以后接入支持上下文的翻译器。
5. **明确锁定的取舍。** 提交过早会把识别错误或不完整语义固定下来；等待更多上下文会增加显示确认时间。已提交块若后来确实发现错误，应显式标记更正，而不是悄悄整体重写。首轮优先保证语义完整和可读性，再缩短确认等待。

仅在前端把目前累计译文强行拆成固定行不能解决问题：不同语言的语序和句子数量不一，新一轮译文不保证拥有相同前缀，机械冻结会丢字、重复或错配。需要后端同时提供块边界、提交状态及按块绑定的最终译文。可先在独立测试页加入块模式，保留累计快照作为调试视图，待真实测试通过后再接入正式“同声传译”页。

稳定前缀提交的参考实践见 [Whisper-Streaming](https://github.com/ufal/whisper_streaming) 的 LocalAgreement。它是可借鉴的策略，当前 Nemotron 采用 RNNT token 流，不能把另一模型的提交规则和延迟数值直接当作本项目的保证。

## 前端接入与约束

正式实现位于 `realtime/api.py`、`realtime/socket.py`、`ui/realtime.*` 和 `ui/audio_transport.js`。Gradio 承载页面，同域前缀为 `/realtime-api`，提供 `meta/start/audio/results/finish/cancel/replay`；`diagnostics` 接收受限的浏览器性能摘要。`meta` 设置 HttpOnly owner cookie，HTTP `audio` 用 base64 小端 PCM16。`GET samples/{index}/audio` 只提供清单内 MP3，浏览器解码后通过同一个 AudioWorklet 播放/上传，音量只作用于声音输出。`replay` 用于服务端静默文件回放，不适合测量浏览器声音与字幕同步。

`start` 经原 HTTP 鉴权后额外返回 `stream_ticket`。建立 `/realtime-api/stream` WebSocket 时，第一条 JSON 发送 `session_id` 与票据；正式前端另带 `asr_updates:true` 协商轻量原文推送，未携带此选项的旧客户端保留原全量结果频率，服务端验证 owner cookie 与单次票据，有效期 30 秒；票据不放在 URL 或诊断日志里。随后每条二进制消息为 **小端 uint32 序号 + 小端 uint32 采样率 + PCM16**，返回 `ack`，不要求每包等待确认。独立推送循环约每 20 ms 检查 ASR 变化，先发 `type=asr` 的轻量消息，snapshot 只含 transcript、asr_version、asr_final、asr_server_time_ms 及首条 ASR/音频计时。较大的 `type=results` 包最短约 100 ms 间隔、最长约 500 ms 心跳，原事件游标和全量快照语义保留。ASR 消息不推进事件游标；前端单独更新原文，并拒绝比已显示 ASR 更旧的结果包或最终标志回退。`ping/pong` 用于 RTT 与时钟偏差估计。音频确认完成后发送 `{"type":"finish","last_chunk_seq":N}`，继续等待最终结果。握手失败但尚未送音频时可以回退原 HTTP；送音频后断开明确结束，不冒险切换并重复输入。

回放默认按真实时钟连续播放，不以 ASR/MT 进度或上传确认水位暂停 AudioContext。可选网络适配模式只按上传积压暂停/恢复，服务端处理队列不控制声音。连续回放上传容量按样本时长预留并受 120 秒上限约束；麦克风及网络适配模式仍保留 8 秒容量，服务端保留原有容量限制，超限明确报错。正式协作者可以移植这个传输层；需保留 owner 鉴权、序号、队列上限和最后音频确认规则。[浏览器 WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

原 `ui/app.js` 依赖 `st-*` 固定 ID 和页面状态，新控件用独立 ID，不能同时启动两套麦克风采集。页面语言下拉使用 source languages，而非把原 Demo 全部语言都当成 Nemotron 可识别语言。

默认 120 秒、20 秒待处理音频、一个会话、64 个确认任务、4096 条事件、最多 8 个缓存会话；超限明确报错。`listening` 无音频/操作约 30 秒后回收，`draining` 不因等待翻译被回收。终态约保留 10 分钟，容量满时较早的终态可提前移除。关闭应用时调用 `close()`。

CPU 检查覆盖真实 processor 特征对齐、短尾帧、中文发布、快 ASR/慢 MT、原文-译文映射、保留检查点、取消、归属与 HTTP；模型质量、显存、真实浏览器与公网延迟通过 Slurm 单独验证。测试方法见配套文档。

## 单 GPU 执行队列

下面的 CUDA stream 优先级适用于原本在同一进程加载的 ASR/MT。vLLM 后端使用另外的进程，此优先级不能保证 ASR 优先于 vLLM；需限制引擎预算，并在共卡测试中检查 ASR 延迟。

真实 ASR/MT 在加载时各自创建 CUDA stream，模型导入和 CPU mock 不创建 GPU 资源。ASR 请求高优先级 -1，文本翻译用另一条 priority=0 流；两者都在加载线程的流上建立一次 `wait_stream` 依赖，之后推理分别进入自己的流，不用全设备同步把二者重新串行。`REALTIME_CUDA_STREAMS=0` 可对照兼容模式。优先级并非硬抢占，改善程度需要 GPU 作业验证；浏览器和网络层的独立推送不应被当作模型推理提速的证据。


## 正式页面与协作接入

公开代码在 `realtime/api.py`、`realtime/socket.py` 和 `ui/realtime.*`、`ui/audio_transport.js`。`ui/endpoint_detector.js` 仅在麦克风会话使用，根据讲话后的连续静音结束输入；先 flush 尾部并等待上传 ACK，再发送 finish。它不切字幕、不提交语义块，也不等待服务端判断静音。

正式 API 前缀为 `/realtime-api`，独立测试仍是 `/rt-test-api`，消息协议相同。所有请求使用同源 cookie；正式 API/WS 同时检查 Gradio 登录和实时 owner，会话与其他浏览器隔离。WebSocket 首条消息仍需 start 返回的单次票据。代理子路径使用相对 URL，不需要把计算节点地址写进前端。

`create_api(service, languages, prefix=..., start_session=..., authorize=...)` 返回可复用的 FastAPI 路由与 lifespan；正式入口通过 `Gradio.launch(app_kwargs=...)` 注册到同一服务器。`InferenceCoordinator` 保护文件任务与实时任务互斥，实时 ASR/MT 之间仍独立调度，不持有跨整个会话的推理锁。`ManagedVLLM` 在正式进程中启动和清理引擎，后端适配器不直接管理服务器。

协作者可以直接复用这些模块，或按现有协议独立实现界面；不应重新实现模型加载和译文版本调度。字幕读取 `translation_display*`，原文优先消费轻量 `type=asr`，停止后继续接收直到 finished。当前没有增加语义块提交协议。
