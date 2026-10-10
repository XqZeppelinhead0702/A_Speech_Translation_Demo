"""Design system and static markup for the Gradio front end.

The visual layer only: app.py still owns every component, event and the inference flow.
style.css holds the design tokens and component styling, app.js the state-driven motion.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import gradio as gr

UI_DIR = Path(__file__).resolve().parent
PRODUCT_NAME = "Speech Translation"

# Status text prefix -> UI state. app.py writes these prefixes into the status box;
# app.js derives the visual state (visualizer, stepper, captions) from them.
UI_STATES = [
    ("正在录音", "recording"),
    ("音频已就绪", "ready"),
    ("排队或处理中", "processing"),
    ("完成", "done"),
    ("处理未完成", "error"),
]
CAPTIONS = {
    "idle": "上传一段音频，或点击麦克风开始说话",
    "recording": "正在聆听… 说完请点击停止",
    "ready": "音频已就绪，可以开始翻译",
    "processing": "正在识别并翻译",
    "done": "翻译完成",
    "error": "未能完成，请查看提示后重试",
}

FONTS_HEAD = """
<meta name="theme-color" content="#F5F3EE" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0B0B0E" media="(prefers-color-scheme: dark)">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Serif+SC:wght@500;600;700&display=swap">
"""

SANS = ["Inter", "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", "Noto Sans SC", "system-ui", "sans-serif"]
MONO = ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"]


def build_theme() -> gr.themes.Base:
    """Base theme mapped onto the design tokens; style.css refines the details."""
    return gr.themes.Base(
        primary_hue="orange", neutral_hue="stone", font=SANS, font_mono=MONO,
        radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_lg,
    ).set(
        body_background_fill="#F5F3EE", body_background_fill_dark="#0B0B0E",
        body_text_color="#131217", body_text_color_dark="#F2F1ED",
        body_text_color_subdued="#6B6872", body_text_color_subdued_dark="#9C99A3",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#151519",
        background_fill_secondary="#FAF9F6", background_fill_secondary_dark="#1B1B20",
        border_color_primary="rgba(19,18,23,0.10)", border_color_primary_dark="rgba(255,255,255,0.09)",
        block_background_fill="transparent", block_background_fill_dark="transparent",
        block_border_width="0px", block_shadow="none", block_padding="0px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_color="#6B6872", block_label_text_color_dark="#9C99A3",
        block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
        block_title_text_color="#6B6872", block_title_text_color_dark="#9C99A3",
        block_title_text_weight="500", block_title_text_size="13px", block_title_padding="0px",
        input_background_fill="#FFFFFF", input_background_fill_dark="#1B1B20",
        input_border_color="rgba(19,18,23,0.10)", input_border_color_dark="rgba(255,255,255,0.09)",
        input_shadow="none", input_shadow_focus="0 0 0 3px rgba(242,84,45,0.18)",
        input_border_color_focus="#F2542D", input_border_color_focus_dark="#FF6A3D",
        button_primary_background_fill="#131217", button_primary_background_fill_dark="#F2F1ED",
        button_primary_background_fill_hover="#2A2930", button_primary_background_fill_hover_dark="#FFFFFF",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0B0B0E",
        button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="rgba(19,18,23,0.05)",
        button_secondary_background_fill_hover_dark="rgba(255,255,255,0.06)",
        button_secondary_text_color="#131217", button_secondary_text_color_dark="#F2F1ED",
        button_secondary_border_color="rgba(19,18,23,0.12)", button_secondary_border_color_dark="rgba(255,255,255,0.12)",
        color_accent="#F2542D", color_accent_soft="rgba(242,84,45,0.10)",
        loader_color="#F2542D", shadow_drop="none", shadow_drop_lg="none",
    )


def load_css() -> str:
    return (UI_DIR / "style.css").read_text(encoding="utf-8")


def load_js() -> str:
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    return (script.replace("__ST_STATES__", json.dumps(UI_STATES, ensure_ascii=False))
            .replace("__ST_CAPTIONS__", json.dumps(CAPTIONS, ensure_ascii=False)))


MARK_SVG = """<svg class="st-mark" viewBox="0 0 28 28" aria-hidden="true">
<rect x="3" y="11" width="3" height="6" rx="1.5"/><rect x="8.5" y="6" width="3" height="16" rx="1.5"/>
<rect x="14" y="9" width="3" height="10" rx="1.5" class="st-mark-accent"/><rect x="19.5" y="4" width="3" height="20" rx="1.5"/>
</svg>"""


def topbar_html(engine_label: str, demo_mode: bool) -> str:
    badge = (
        '<span class="st-pill st-pill-demo" title="不运行模型，只返回内置示例的数据集参考文本">'
        '<span class="st-pill-dot"></span>演示模式</span>'
        if demo_mode else
        f'<span class="st-pill" title="模型已在服务器本地加载"><span class="st-pill-dot st-dot-ok"></span>{escape(engine_label)}</span>'
    )
    return f"""<header class="st-topbar">
  <a class="st-brand" href="./" aria-label="{PRODUCT_NAME}">{MARK_SVG}<span>{PRODUCT_NAME}</span></a>
  <div class="st-topbar-right">{badge}</div>
</header>"""


def hero_html(language_count: int, demo_mode: bool) -> str:
    notice = (
        '<p class="st-demo-note"><strong>演示模式</strong>不加载模型，内置示例返回 CoVoST 2 数据集的参考原文与译文，'
        '用于预览界面；正式演示请在 GPU 环境运行完整模型。</p>'
        if demo_mode else ""
    )
    return f"""<section class="st-hero">
  <p class="st-eyebrow">语音识别 · 自动语种检测 · {language_count} 种语言互译</p>
  <h1><span class="st-clause">开口说话，</span><span class="st-clause"><em>即刻</em>跨越语言。</span></h1>
  <p class="st-lede">上传或录制一段语音，自动判断说话语言，同时给出原文与译文。</p>
  {notice}
</section>"""


VISUALIZER_HTML = f"""<div class="st-viz" aria-hidden="true">
  <canvas id="st-viz-canvas"></canvas>
</div>
<p id="st-viz-caption" class="st-viz-caption" role="status" aria-live="polite">{CAPTIONS["idle"]}</p>"""

STEPPER_HTML = """<ol class="st-steps" aria-label="处理进度">
  <li data-step="audio"><span class="st-step-dot"></span>音频</li>
  <li data-step="lang"><span class="st-step-dot"></span>语种</li>
  <li data-step="asr"><span class="st-step-dot"></span>识别</li>
  <li data-step="mt"><span class="st-step-dot"></span>翻译</li>
</ol>"""


def hint_html(max_seconds: float) -> str:
    return f"""<p class="st-hint">单段最长 {max_seconds:g} 秒 · 麦克风录音需 HTTPS 与浏览器授权<br>
<kbd>Ctrl</kbd> / <kbd>⌘</kbd> + <kbd>Enter</kbd> 提交</p>"""


def section_head_html(title: str, note: str = "") -> str:
    note_html = f'<span class="st-section-note">{escape(note)}</span>' if note else ""
    return f'<div class="st-section-head"><h2>{escape(title)}</h2>{note_html}</div>'


INTERPRET_HTML = """<section class="st-soon">
  <p class="st-eyebrow"><span class="st-soon-badge">即将推出</span></p>
  <h1>同声传译</h1>
  <p class="st-lede">边说边译：在说话的同时持续输出双语字幕，适合会议、访谈与跨语言交流。</p>
  <ul class="st-soon-grid">
    <li><span class="st-soon-icon">01</span><h3>流式识别</h3><p>语音分段送入模型，字幕随说话实时出现。</p></li>
    <li><span class="st-soon-icon">02</span><h3>双语对照</h3><p>原文与译文逐句对齐，回看与复制都很方便。</p></li>
    <li><span class="st-soon-icon">03</span><h3>共享语言与模型</h3><p>沿用当前 21 种语言和本地部署方式。</p></li>
  </ul>
  <p class="st-soon-foot">该功能仍在规划中，当前版本不提供实时传译。</p>
</section>"""


def footer_html(engine_label: str) -> str:
    return f"""<footer class="st-footer">
  <span>{escape(engine_label)}</span><span aria-hidden="true">·</span>
  <span>临时音频约 1 小时后自动清理</span><span aria-hidden="true">·</span><span>结果可能有误，重要内容请人工核对</span>
</footer>"""
