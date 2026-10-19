# 示例音频来源

[返回测试说明](../../docs/realtime_testing.md#展示样本与观察方法)

## 当前展示：10 条 10–30 秒真人朗读

文件翻译页和实时测试页统一读取 [realtime_samples.json](realtime_samples.json)，只展示下表 10 条节选。覆盖 8 种输入语言、10 组默认翻译方向，均在 Nemotron ASR、SeamlessM4T 和 Hy-MT2 现有支持范围内。单声道、16 kHz、48 kbps MP3，合计约 1.21 MiB；音频随项目提供，使用时无需下载原章。

| 文件 | 默认方向 | 实际时长 | 作品 / 章节 | 朗读者 | 来源 |
| --- | --- | --- | --- | --- | --- |
| [demo_01_cmn_eng.mp3](demo_01_cmn_eng.mp3) | 中文 → 英语 | 18.2 秒 | 孔乙己 / Kong Yiji | Jing Li | [目录](https://librivox.org/call-to-arms-by-xun-lu/) · [原录音](https://archive.org/download/call_to_arms_jl_librivox/calltoarms_03_lu_64kb.mp3) · [作品原文](https://zh.wikisource.org/wiki/孔乙己) |
| [demo_02_cmn_jpn.mp3](demo_02_cmn_jpn.mp3) | 中文 → 日语 | 25.0 秒 | 孔乙己 / Kong Yiji | Jing Li | [目录](https://librivox.org/call-to-arms-by-xun-lu/) · [原录音](https://archive.org/download/call_to_arms_jl_librivox/calltoarms_03_lu_64kb.mp3) · [作品原文](https://zh.wikisource.org/wiki/孔乙己) |
| [demo_03_eng_cmn.mp3](demo_03_eng_cmn.mp3) | 英语 → 中文 | 15.6 秒 | A Scandal in Bohemia (The Adventures of Sherlock Holmes, version 4) | David Clarke | [目录](https://librivox.org/9481) · [原录音](https://archive.org/download/adventuressherlockholmes_v4_1501_librivox/adventuresofsherlockholmes_01_doyle_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/1661) |
| [demo_04_eng_fra.mp3](demo_04_eng_fra.mp3) | 英语 → 法语 | 27.3 秒 | A Scandal in Bohemia (The Adventures of Sherlock Holmes, version 4) | David Clarke | [目录](https://librivox.org/9481) · [原录音](https://archive.org/download/adventuressherlockholmes_v4_1501_librivox/adventuresofsherlockholmes_01_doyle_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/1661) |
| [demo_05_fra_cmn.mp3](demo_05_fra_cmn.mp3) | 法语 → 中文 | 20.1 秒 | La Bécasse (Les Contes de la bécasse) | Didier | [目录](https://librivox.org/les-contes-de-la-becasse-by-guy-de-maupassant/) · [原录音](https://archive.org/download/les_contes_de_la_becasse_0809_librivox/lescontesdelabecasse_01_maupassant_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/11714) |
| [demo_06_deu_eng.mp3](demo_06_deu_eng.mp3) | 德语 → 英语 | 18.4 秒 | Die Verwandlung / Abschnitt 1 | Markus Wachenheim | [目录](https://librivox.org/die-verwandlung-von-franz-kafka/) · [原录音](https://archive.org/download/verwandlung_mw_0811_librivox/verwandlung_01_kafka_64kb.mp3) · [作品原文](https://www.zeno.org/Literatur/M/Kafka,+Franz/Erzählungen+und+andere+Prosa/Die+Verwandlung) |
| [demo_07_spa_cmn.mp3](demo_07_spa_cmn.mp3) | 西班牙语 → 中文 | 21.6 秒 | Don Quijote 1 / Capítulo I–III | Nessi | [目录](https://librivox.org/don-quijote-vol-1-by-miguel-de-cervantes-saavedra/) · [原录音](https://archive.org/download/don_quijote_vol1_0706_librivox/quijote_vol1_03_cervantes_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/2000) |
| [demo_08_ita_deu.mp3](demo_08_ita_deu.mp3) | 意大利语 → 德语 | 23.1 秒 | Le Avventure di Pinocchio / Chapter 01 | Simona | [目录](https://librivox.org/le-avventure-di-pinocchio-by-c-collodi/) · [原录音](https://archive.org/download/avventure_pinocchio_librivox/avventurepinocchio_01_collodi_64kb.mp3) · [作品原文](https://it.wikisource.org/wiki/Le_avventure_di_Pinocchio) |
| [demo_09_por_spa.mp3](demo_09_por_spa.mp3) | 葡萄牙语 → 西班牙语 | 24.4 秒 | Dom Casmurro / Capítulos I a V | Leni | [目录](https://librivox.org/dom-casmurro-by-machado-de-assis/) · [原录音](https://archive.org/download/dom_casmurro_2102_librivox/domcasmurro_01_assis_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/55752) |
| [demo_10_nld_eng.mp3](demo_10_nld_eng.mp3) | 荷兰语 → 英语 | 15.8 秒 | De Kleine Johannes / Hoofdstuk 01 | Bart de Leeuw | [目录](https://librivox.org/de-kleine-johannes-door-frederik-van-eeden/) · [原录音](https://archive.org/download/kleine_johannes_0909_librivox/johannes_01_eeden_64kb.mp3) · [作品原文](https://www.dbnl.org/tekst/eede003klei01/) |

每条是连续真人录音节选，没有拼接短句、重复内容或生成合成语音。起止点选在检测到的短停顿中（阈值 -35 dB、至少 0.28 秒），减少裁到词中间的情况；停顿不等于句末，这不是逐句人工转写或语义对齐。中文、英文各提供两段不同节选，其他语言各一段。选择样本会填入其输入语言和建议输出语言，也可自行切换到其他受支持目标语言。

统一采样率和轻量音量校正后，用 FFmpeg `libmp3lame` 编码，关闭 bit reservoir（`-reservoir 0`），保留当前浏览器和 SoundFile 的回放兼容性。清单记录原章裁剪时间、音量增益、SHA-256、作者、朗读者和来源链接。中英法片段从原有 90 秒本地文件裁剪，`derived_from` 记录该文件，时间区间仍按上游完整原章计。

这些节选没有逐字对齐的参考转写或标准参考译文；原文链接只供人工查阅。`references.csv` 仍仅覆盖旧 CoVoST 短样本。不要把整章文本用于计算节选 CER，CPU 演示模式也不会为新节选伪造识别或翻译结果。

录音来源为 LibriVox；上游标注为 **Public Domain in the USA**，见 [LibriVox 录音复用说明](https://librivox.org/pages/public-domain/)。作者与朗读者信息按原目录保留。

## 90 秒连续朗读样本

以下文件已随项目提供，无需下载完整有声书。每段来自一位朗读者的连续录音，截取原章第 **45–135 秒**，没有重复、拼接短句或合成语音。统一转换为单声道、16 kHz、48 kbps MP3，每个约 528 KiB，合计约 1.55 MiB。裁剪边界可能位于句中，适合观察持续输入及尾段处理。

| 本地文件 | 作品 / 章节 | 作者 | 朗读者 | 原录音与文本 |
| --- | --- | --- | --- | --- |
| [zh_librivox_90s.mp3](zh_librivox_90s.mp3) | 《呐喊》/《孔乙己》 | 鲁迅 | Jing Li | [LibriVox](https://librivox.org/call-to-arms-by-xun-lu/) · [原录音](https://archive.org/download/call_to_arms_jl_librivox/calltoarms_03_lu_64kb.mp3) · [章节原文](https://zh.wikisource.org/wiki/孔乙己) |
| [en_librivox_90s.mp3](en_librivox_90s.mp3) | The Adventures of Sherlock Holmes (version 4) / A Scandal in Bohemia | Arthur Conan Doyle | David Clarke | [LibriVox](https://librivox.org/9481) · [原录音](https://archive.org/download/adventuressherlockholmes_v4_1501_librivox/adventuresofsherlockholmes_01_doyle_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/1661) |
| [fr_librivox_90s.mp3](fr_librivox_90s.mp3) | Les Contes de la bécasse / La Bécasse | Guy de Maupassant | Didier | [LibriVox](https://librivox.org/les-contes-de-la-becasse-by-guy-de-maupassant/) · [原录音](https://archive.org/download/les_contes_de_la_becasse_0809_librivox/lescontesdelabecasse_01_maupassant_64kb.mp3) · [作品原文](https://www.gutenberg.org/ebooks/11714) |

上游将录音标注为 **Public Domain in the USA**，录音复用说明见 [LibriVox Public Domain](https://librivox.org/pages/public-domain/)。此处保留作者、朗读者和来源信息；这不是本项目自行录制的音频。

[realtime_samples_long.json](realtime_samples_long.json) 记录语言、时长、裁剪区间、编码参数、来源和本地文件 SHA-256，保留旧长样本的来源信息；这些音频不再进入展示列表。裁剪使用 FFmpeg 的 `-ss 45 -t 90 -map_metadata -1 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k -reservoir 0` 参数；先解码再裁剪，保证有效采样时长为 90 秒。关闭 MP3 bit reservoir，避免当前 SoundFile/libsndfile 在逐块读取后的内部 seek 产生解码警告。上游完整文件只需在需要重新制作样本时下载，运行 Demo 不会联网获取数据。

章节原文用于人工核对内容，**不是与这 90 秒逐字对齐的参考转写**，也没有提供标准参考译文。不要直接把整章文本传给 `--reference-text` 计算 CER；需要精度指标时，请先人工听写对应节选。

## 原有短样本

`zh-CN_en_*.mp3`、`en_zh-CN_*.mp3`、`fr_en_*.mp3` 来自 [fixie-ai/covost2](https://huggingface.co/datasets/fixie-ai/covost2)，对应原文与参考译文见 [references.csv](references.csv)。它们不再出现在任何展示样本列表，文件仍保留，可手动上传或通过命令行做短句和尾段检查。`references.csv` 仅覆盖这些原有短样本，不覆盖新增的 LibriVox 节选。
