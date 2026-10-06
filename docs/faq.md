# Q&A、注意事项与排错

[返回 README](../README.md)

## 只显示 `Running on local URL: http://0.0.0.0:7860`，怎么从外部访问？

这是监听信息，不是公网地址。临时分享应等待实际 `PUBLIC_URL=https://….gradio.live`；登录节点中转模式中的 URL 由登录节点进程打印。固定域名模式不生成 Gradio URL，应访问自己配置的 HTTPS 域名。完整过程见[网络说明](network.md)。

## Gradio API 报 `ConnectTimeout` 怎么办？

先确认是哪个节点上的进程无法联网。登录终端可联网不代表计算节点可联网：

- 计算节点有出口：检查 DNS、防火墙和代理，使用 `serve_direct.sh`。
- 只有登录节点有出口：使用 `serve_login_relay.sh`，显式配置登录节点的 `DEMO_RELAY_PROXY_URL`。
- 无法连接 Gradio，但能 SSH 到公网服务器：使用 `serve_public_tunnel.sh` 和自己的域名入口。

HTTP API 能成功不等于分享连接也能建立；使用完整的 `--check-only --test-tunnel` 验证。回环代理的 `127.0.0.1` 表示当前运行进程所在机器，在登录节点存在的代理不一定在计算节点存在。

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
python -m py_compile app.py scripts/check_public_access.py scripts/login_relay.py
bash -n scripts/run/serve.sh
python app.py --help
```

公网检查工具也不加载模型，但会联网、下载客户端或建立诊断连接。真正启动服务的脚本会加载 GPU 模型；在集群上务必通过 Slurm 提交。
