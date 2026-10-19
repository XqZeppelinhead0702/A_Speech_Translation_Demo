# Q&A、注意事项与排错

[返回 README](../README.md)

## 默认两页需要装哪些新增依赖？

默认部署使用 `requirements-vllm.txt`，其中包含基础依赖、实时音频/HTTP/WebSocket 依赖和 vLLM 0.10.2；先安装与驱动匹配的 Torch/Torchaudio CUDA 构建。已有能正常运行的环境直接沿用，启动脚本不会安装包。仅文件端到端模式无需 vLLM，完整组合见[环境说明](environment.md#环境基线与依赖选择)。

`pip check` 只能检查包声明依赖；vLLM 的旧 tokenizer 接口还需要项目自带的局部兼容层。不要只复制 `app.py` 而遗漏 `scripts/vllm_compat/` 或 `realtime/`。运行前可用 `bash scripts/run/serve_vllm.sh --check-only` 检查版本和本地配置，不加载权重。

## 试听没有自动播放，或实时录音提前结束

文件提交后的试听不影响识别/翻译，浏览器拒绝自动播放时手动点击试听播放器即可。麦克风默认在讲话后静音 2.5 秒自动结束；停顿较多时选择 4 秒或关闭自动结束，再手动停止。检测依据浏览器音频能量，轻声与持续背景噪声会影响判断，不保证语义句子已经完整。示例回放不使用该检测。见[页面操作](formal_demo.md#演示播放与录音自动结束)。

## 录音后提示没有音频，或重复点击导致状态混乱

麦克风录音要先点击停止，再等状态显示“音频已就绪”后点击“识别并翻译”。本地出现波形不一定代表录音已传到服务器；公网中转较慢时，音频编码、上传可能还在进行。[Gradio 录音上传流程](https://github.com/gradio-app/gradio/blob/main/js/audio/interactive/InteractiveAudio.svelte)

页面在音频值准备好之前禁用提交，录音开始时重新禁用；排队和处理期间会禁用当前页面的提交、清空、输入切换及示例，完成或出错后恢复。就绪提示和控件切换在浏览器执行，不额外等待服务器请求。其他用户仍使用各自的音频，模型推理仍串行执行，队列最多 16 个请求。

如果一直没有就绪提示，检查麦克风授权和网络连接，再重新录音或上传文件；不要只反复点击按钮。服务重启后旧页面可能持有失效的缓存引用，需要刷新并重新输入。程序不会用上一段音频代替空输入，避免无意中翻译旧录音。

## 自动语言检测不正确或提示置信度不足

输入下拉框保留手动选择，可改为音频实际语言后重试；手动模式不执行检测推理。短音频、口音、背景噪声和多语言混说可能造成误判，高置信度也不代表一定正确。默认 MMS 含粤语类别；切回 SpeechBrain 后，粤语需手动指定。普通话自动识别默认使用简体原文。检测至少需要 1 秒有效音频，建议提供更长、清晰的单语片段。配置、置信度阈值及检测过程见[自动语言检测](language_detection.md)。

## 缺少语言检测模型或依赖

先确认当前 `DEMO_LID_BACKEND`：默认 `mms` 需要 `ckpts/mms-lid-256/` 中的 `config.json`、`preprocessor_config.json` 和分类权重；`speechbrain` 需要 `ckpts/lang-id-voxlingua107-ecapa/` 中的 `hyperparams.yaml`、`embedding_model.ckpt`、`classifier.ckpt`、`label_encoder.txt`。服务实际使用的 Python 环境需安装 `requirements.txt`。其他位置通过 `DEMO_LID_MODEL_DIR` 或 `--lid-model-dir` 指定，切换模块时必须匹配目录；已有配置若固定了 SpeechBrain 路径，使用默认 MMS 前需删除或修改该路径设置。程序只读取本地权重，不会在无法联网的计算节点自动下载。

## 只显示 `Running on local URL: http://0.0.0.0:7860`，怎么从外部访问？

这是监听信息，不是公网地址。临时分享应等待实际 `PUBLIC_URL=https://….gradio.live`；登录节点中转模式中的 URL 由登录节点进程打印。固定域名模式不生成 Gradio URL，应访问自己配置的 HTTPS 域名。完整过程见[网络说明](network.md)。

## Gradio API 报 `ConnectTimeout` 怎么办？

先确认是哪个节点上的进程无法联网。登录终端可联网不代表计算节点可联网：

- 计算节点有出口：检查 DNS、防火墙和代理，使用 `serve_direct.sh`。
- 只有登录节点有出口：使用 `serve_login_relay.sh`，显式配置登录节点的 `DEMO_RELAY_PROXY_URL`。
- 无法连接 Gradio，但能 SSH 到公网服务器：使用 `serve_public_tunnel.sh` 和自己的域名入口。

HTTP API 能成功不等于分享连接也能建立；使用完整的 `--check-only --test-tunnel` 验证。回环代理的 `127.0.0.1` 表示当前运行进程所在机器，在登录节点存在的代理不一定在计算节点存在。

## FRPC 报 `login to server failed: EOF` / `session shutdown`

这表示创建 Gradio 公网分享连接时，握手连接被关闭。若日志已经显示实际运行节点和 API/TCP 检查通过，SSH 已将远端检查程序启动；应检查登录节点到 Gradio 分享服务器的网络，而不是仅根据旧版通用提示重新配置 SSH 密钥。

API 可访问、代理响应 TCP CONNECT 成功，仍不保证 TLS/FRPC 握手能完成。可能是代理链路中断、网络限制或分享服务暂时异常，仅凭这句错误不能确认是哪一种。

如果测试发现分享服务器直连成功、经代理失败，而 API 仍需代理，可以在 `.env.local` 中保留原来的代理，并增加一行：

```bash
DEMO_RELAY_PROXY_URL=http://127.0.0.1:7890
DEMO_RELAY_FRPC_PROXY_URL=
```

代理端口应使用登录节点实际配置。显式空值让 FRPC 直连；不设置该变量时，FRPC 沿用原来的代理。完整含义见[中转配置](configuration.md#登录节点中转)。若登录节点也不能直接连接分享服务器，这个设置无法解决问题，需要修复代理或使用另一个可用出口。

修改后先做不加载 GPU 模型的完整检查；诊断端口需选择一个空闲端口，例如 `27860`：

```bash
source scripts/run/common.sh
"$DEMO_PYTHON" -u scripts/login_relay.py connect \
  --target "$DEMO_RELAY_TARGET" --relay-port 27860 \
  --python "${DEMO_RELAY_PYTHON:-$(command -v "$DEMO_PYTHON")}" \
  --project-dir "${DEMO_RELAY_PROJECT_DIR:-$PWD}" \
  --check-only --test-tunnel
```

出现“公网隧道握手通过；诊断隧道立即关闭”后再重新提交 GPU 作业。诊断 URL 随测试结束失效，不能用作实际 Demo 入口。

## FRPC 报 `lookup gradio-live.com: i/o timeout`

这是建立公网隧道的 FRPC 客户端在登录节点上解析分享服务器域名超时，发生在模型和翻译逻辑之外。仅凭这一条日志不能确定是临时网络波动还是节点 DNS 长期异常。Gradio 通过独立的 FRPC 进程连接分享服务器，因此 Python API/TCP 检查通过后，客户端后续解析仍可能失败。[Gradio 隧道实现](https://github.com/gradio-app/gradio/blob/main/gradio/tunneling.py)

中转脚本对这类临时 DNS 错误最多尝试 3 次，每次间隔 2 秒；每次失败都会关闭失败的 FRPC 进程。正常连接成功时仍只尝试一次，SSH 目标、API 代理、FRPC 直连/代理设置、证书校验和公网入口保持原样。`EOF` / `session shutdown` 等其他错误不会触发这项重试。

先用当前 `.env.local` 做无 GPU 检查，无需先提交模型作业；诊断端口选空闲值：

```bash
source scripts/run/common.sh
"$DEMO_PYTHON" -u scripts/login_relay.py connect \
  --target "$DEMO_RELAY_TARGET" --relay-port 27862 \
  --python "${DEMO_RELAY_PYTHON:-$(command -v "$DEMO_PYTHON")}" \
  --project-dir "${DEMO_RELAY_PROJECT_DIR:-$PWD}" \
  --check-only --test-tunnel
```

看到“公网隧道握手通过；诊断隧道立即关闭”后，重新提交原来的服务作业即可。诊断链接随检查结束失效。原 Demo 和实时测试入口共用同一个中转脚本，均获得这项重试，不需要分别修改。

如果连续失败，在日志所示的**实际登录节点**运行：

```bash
timeout 20s getent ahostsv4 gradio-live.com
```

没有地址输出或经常超时，请向管理员反馈节点名、报错时间和域名，检查该节点 DNS 服务。如果系统解析成功而 FRPC 持续失败，也反馈完整 FRPC 错误；一次系统查询成功不能保证独立客户端的下一次查询成功。不要为此重配 SSH 密钥、修改 GPU 参数或把分享服务器 IP 写死：地址可能变化，直接替换为 IP 还可能导致 TLS 域名校验失败。

## `No module named 'scripts.check_public_access'`

登录节点已有项目文件，但 Python 导入了环境中另一个同名 `scripts` 包时，也会报这个错误。项目现在包含 `scripts/__init__.py`，配合中转入口把项目根目录放在 `sys.path` 首位，让它作为普通包优先导入。这个问题发生在公网检查之前，无需为此修改代理或 SSH 密钥。

确认登录节点使用的项目副本也包含 `scripts/__init__.py` 和 `scripts/check_public_access.py`；若 `DEMO_RELAY_PROJECT_DIR` 指向另一个副本，同步更新该副本再提交。无需卸载提供同名包的第三方依赖。可以在项目根目录运行 CPU 导入检查：

```bash
python -c "import scripts, scripts.check_public_access as c; print(scripts.__file__); print(c.__file__)"
```

两条路径都应指向本项目 `scripts/`，不应指向 `site-packages/scripts/`。之后再使用前面的 `--check-only --test-tunnel` 做无模型中转检查。

## `No ... host key is known` / `Host key verification failed`

需要在**运行作业的集群账号**下核对主机指纹并建立 `~/.ssh/known_hosts` 记录。个人电脑上已有的记录不会自动传给集群。保持 `StrictHostKeyChecking=yes`；准备步骤见 [Slurm SSH 配置](deploy_slurm.md#2-准备免交互-ssh)。

## `/tmp/gradio/vibe_edit_history` 权限错误

入口在导入 Gradio 前设置专属可写目录。检查是否自行设置了不可写的 `GRADIO_TEMP_DIR`，可删除对应配置、使用默认路径，或设置为自己的可写目录。默认缓存位置为项目 `.cache/gradio/` 下的作业/进程目录。

## 缓存能否删除？

可以删除**已经停止的实例**的缓存。不要删除运行中实例正在使用的目录，否则上传/音频转换和页面请求可能失败。

| 路径 | 内容与处理 |
| --- | --- |
| `.cache/gradio/job-<作业号>/` | 作业的临时文件，作业停止后可清理 |
| `.cache/gradio/relay-<作业号或进程号>/` | 登录节点中转临时文件，中转停止后可清理 |
| `.cache/gradio/process-<进程号>/` | 普通实例临时文件，进程停止后可清理 |
| 自己设置的 `GRADIO_TEMP_DIR` | 确认实际路径及使用它的实例均停止后清理 |
| `.gradio/` | Gradio 生成文件，停服后可清理，可能需重新下载 |
| `~/.cache/huggingface/gradio/` | 用户级分享客户端，删除后可能需要联网重新准备 |
| vLLM / Torch 编译缓存 | 不属于录音缓存；清理前停止对应引擎，下次启动可能重新编译。不要为清理上传文件直接清空用户级缓存 |
| `ckpts/` | 模型文件，不属于可以随意清空的应用缓存 |
| `data/test_samples/` | 自带示例，不属于缓存，应保留 |
| `outs/` | 日志，与缓存独立，可在无需排错/保留时另行清理 |

从项目根目录查看用量：

```bash
du -sh .cache/gradio .gradio outs 2>/dev/null
ls -1 .cache/gradio
```

确认旧作业及其中转都已停止后，可只删除对应缓存：

```bash
# 123456 为示例旧作业号。
squeue -j 123456
rm -rf -- .cache/gradio/job-123456 .cache/gradio/relay-123456
```

只有**所有相关 Demo 与中转都停止**后，才可以清空全部项目 Gradio 缓存：

```bash
# 在项目根目录执行。
rm -rf -- .cache/gradio
```

下次启动会重新创建专属目录。Gradio 的 `delete_cache=(3600, 3600)` 按一小时周期清理符合条件且超过一小时的临时文件，不保证清除所有目录。无需删除整个 `~/.cache`，也不要把模型或 SSH 文件当作临时缓存删除。

## 可以一直运行，直到主动停止吗？

普通机器可以通过 systemd 持续运行；Slurm 上程序不主动计时退出，但分区/QOS 的时限、抢占或故障仍可能结束作业。无限作业需要集群允许，详见 [Slurm 时限](deploy_slurm.md#持续运行与时限)。Gradio 临时分享通常一周到期，需要长期固定入口时使用域名方案。

## 固定域名返回 502

先等模型加载完，再检查应用是否在运行、SSH 转发是否正常、Caddy 后端端口是否匹配。代理在应用尚未就绪时可能短暂返回 502。同机部署通常代理 7860，公网服务器中转通常代理 17860。

## 麦克风不能录音

使用实际 HTTPS URL，并授权浏览器麦克风权限。普通远程 HTTP 地址通常无法录音；本机 localhost 是常见例外。

## 有很多显卡，是否自动使用全部 GPU？

单个实例将模型放在第一张可见 GPU，一次执行一个任务。可设置 `CUDA_VISIBLE_DEVICES=0` 指定设备。多个实例可以分别设置可见 GPU 和端口，但每个实例会加载一份权重；当前版本没有多 GPU 分片或调度机制。

## 权重找不到、音频解码失败或显存不足

检查完整模型路径、FFmpeg 和 PyTorch CUDA 环境；显存不足时先缩短音频。推荐清晰、30 秒以内的单语片段。默认最大 60 秒、25 MB，不应把显存或时长上限理解为质量保证。

## 如何在不运行 GPU 模型的情况下检查？

```bash
python -m py_compile app.py language_detection.py scripts/check_public_access.py scripts/login_relay.py
bash -n scripts/run/serve.sh
python app.py --help
```

公网检查工具也不加载模型，但会联网、下载客户端或建立诊断连接。真正启动服务的脚本会加载 GPU 模型；在集群上务必通过 Slurm 提交。


## 文件级联一直显示“正在转写”

“正在转写”表示还没有进入 Hy-MT2/vLLM 翻译阶段。旧实现使用 SeamlessM4T 整段转写，并等整次生成结束后才显示原文；不能据此判断 vLLM 或公网入口故障。

当前文件级联改为 Nemotron 官方非流式 API，复用实时页已加载的 ASR，显示原文进度和耗时；端到端选项继续使用 SeamlessM4T。未覆盖的文件输入语言保留 SeamlessM4T 兼容路径。检查新的作业日志：

- `File ASR started`：实际选择的 ASR、模式、语言、音频时长。
- `File ASR completed`：转写完成与耗时；无此行时，问题仍在 ASR 阶段。
- `File translation started`：已拿到最终原文并开始文本翻译，此后再检查 vLLM 日志。

默认 `DEMO_FILE_ASR_TIMEOUT=120`，在可执行的生成步骤和输出回调处检查超时，异常不会继续提交翻译。Python 无法强行抢占已经卡住的 CUDA 调用，遇到底层 GPU 错误仍需停止并重提作业；错误日志用于定位具体调用。更新后需重提 Slurm 作业，刷新旧页面不会更新正在运行的 Python 代码。
