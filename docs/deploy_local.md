# 普通 GPU 机器与固定域名部署

[返回 README](../README.md) · [网络原理](network.md) · [配置项](configuration.md)

本页用于可以直接使用 GPU 的 Linux 机器，无需 Slurm。首先按[环境说明](environment.md)安装依赖、准备完整模型，进入项目根目录，激活环境并执行 `cp -n .env.example .env.local`。

## A. 临时公网链接

```bash
bash scripts/run/serve_share.sh
```

脚本先检查公网，再启动 GPU 模型。等待 `PUBLIC_URL=https://….gradio.live`，即可在其他电脑或手机访问。无需开放机器的 7860 入站端口；机器需要能连接 Gradio API 和分享服务器。停止用 `Ctrl+C`。

需要断开终端后继续测试时，可在 `tmux` 中运行。临时分享依赖进程和网络，通常一周到期，重启会换地址。[Gradio 分享说明](https://gradio.app/guides/sharing-your-app)

## B. 仅本机或局域网使用

```bash
bash scripts/run/serve_local.sh
```

默认访问 `http://127.0.0.1:7860`，仅运行服务的机器能打开。需要局域网其他设备访问时，在 `.env.local` 设置 `DEMO_HOST=0.0.0.0`，并允许本机 7860 入站，再访问机器真实局域网 IP；普通远程 HTTP 页面通常无法使用麦克风。

局域网监听不会自动建立公网入口。公网使用下面的 HTTPS 方案。

## C. GPU 机器本身有公网入口：固定域名

需要域名 DNS 指向该机器的公网 IP，并允许公网 TCP 80/443。路由器后的机器还需将这两个端口转发到它；若宽带没有可用入站公网地址，使用下一节的公网服务器中转。

按照 [Caddy 官方步骤](https://caddyserver.com/docs/install)安装 Caddy。后端保持 `.env.local` 的 `DEMO_HOST=127.0.0.1`，在一个终端执行：

```bash
bash scripts/run/serve_local.sh
```

在同机另一个终端执行以下 HTTPS 入口脚本，将域名替换为实际值：

```bash
sudo bash scripts/deploy/run_https_proxy.sh speech.example.com 7860
```

这个脚本生成临时 Caddy 配置并前台运行代理，不改写已有配置文件。需要 80/443 绑定权限；如果已经有 Caddy/其他 Web 服务占用这些端口，应把下述配置加入现有服务，而不是启动第二个代理。

```caddyfile
speech.example.com {
    reverse_proxy 127.0.0.1:7860
}
```

模型和代理均就绪后访问 `https://自己的域名`。Caddy 在 DNS/网络条件满足时维护 HTTPS 证书，7860 保持本机访问即可。[Caddy HTTPS 说明](https://caddyserver.com/docs/quick-starts/https)

## 公网服务器中转

适合 GPU 机器没有公网入站入口，但能够 SSH 到一台公网服务器的场景；这台公网服务器不需要 GPU。

### 1. 公网服务器准备 HTTPS 入口

在公网服务器安装 Caddy，将域名 DNS 指向它并开放 80/443。复制 `scripts/deploy/run_https_proxy.sh` 到公网服务器运行，或让服务器可以访问这份脚本：

```bash
# 在公网服务器执行，17860 是 SSH 转发接收端口。
sudo bash scripts/deploy/run_https_proxy.sh speech.example.com 17860
```

该命令对应仓库 `scripts/deploy/Caddyfile` 的配置。**同机部署使用 7860，公网服务器中转使用 17860**，修改端口时同步修改相关配置。

### 2. GPU 机器准备 SSH

在 GPU 机器运行账号的 `~/.ssh/config` 配置，例如：

```sshconfig
Host speech-demo-public
    HostName your-public-server
    User demo
    Port 22
    IdentityFile ~/.ssh/speech_demo_public
    IdentitiesOnly yes
```

将专用密钥公钥安装到公网服务器，首次登录核对指纹，确保 `ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes speech-demo-public true` 成功。公网服务器 SSH 必须允许 remote TCP forwarding，且本机 17860 未被占用。

### 3. 启动后端和转发

在 GPU 机器的 `.env.local` 设置：

```bash
DEMO_TUNNEL_TARGET=speech-demo-public
DEMO_TUNNEL_PORT=17860
```

然后执行：

```bash
bash scripts/run/serve_public_tunnel.sh
```

脚本先检查 SSH 认证，再启动模型和反向转发，访问 `https://自己的域名` 即可。后端和转发由同一个脚本管理；任一进程退出会结束另一个。`Ctrl+C` 同时关闭它们。公网服务器上的 Caddy 也需保持运行。

## 长期运行：systemd

仓库提供 `scripts/deploy/speech-demo.service.example`。模板默认使用本机后端入口；需要公网服务器转发时，将 `ExecStart` 的入口改为 `serve_public_tunnel.sh`。

先按实际位置修改模板中的用户、项目目录和脚本路径，并在项目 `.env.local` 中配置 `DEMO_PYTHON` 或 Conda 初始化、FFmpeg 及模型路径。systemd 不执行交互式 `conda activate`；运行账号需能读取模型、写缓存并使用 GPU。

将填写好的配置安装为系统服务：

```bash
sudo install -m 644 /path/to/edited-speech-demo.service /etc/systemd/system/speech-demo.service
sudo systemctl daemon-reload
sudo systemctl enable --now speech-demo
sudo systemctl status speech-demo
sudo journalctl -u speech-demo -f
sudo systemctl stop speech-demo
```

固定域名入口也要持续运行：使用已安装 Caddy 的 systemd 服务，把对应后端配置加入 `/etc/caddy/Caddyfile`，保留其他已有站点，然后：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

普通机器同机后端为 7860；公网服务器中转后端为 17860。需要关闭应用开机启动时使用 `sudo systemctl disable speech-demo`。
