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
# app.js derives the visual state (voice wave, status island, glow) from them.
UI_STATES = [
    ("正在录音", "recording"),
    ("音频已就绪", "ready"),
    ("排队或处理中", "processing"),
    ("完成", "done"),
    ("处理未完成", "error"),
]
# Status island copy per state; app.js appends live timers where noted.
ISLAND = {
    "idle": "上传音频，或点按麦克风开始",
    "recording": "正在聆听",
    "ready": "音频已就绪",
    "processing": "正在识别与翻译",
    "done": "翻译完成",
    "error": "未能完成",
}

FONTS_HEAD = """
<meta name="theme-color" content="#FBFBFD" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#000000" media="(prefers-color-scheme: dark)">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<style>
/* Registered here because Gradio's custom-CSS processing drops @property rules.
   Lets the spectrum glow rotate smoothly via a keyframed angle. */
@property --st-angle { syntax: "<angle>"; initial-value: 0deg; inherits: false; }
</style>
"""

# Native first: SF Pro + PingFang on Apple devices, Inter + YaHei / Noto elsewhere.
SANS = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "PingFang SC",
        "HarmonyOS Sans SC", "Microsoft YaHei", "Noto Sans SC", "system-ui", "sans-serif"]
MONO = ["SF Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"]


def build_theme() -> gr.themes.Base:
    """Base theme mapped onto the design tokens; style.css refines the details."""
    return gr.themes.Base(
        primary_hue="violet", neutral_hue="gray", font=SANS, font_mono=MONO,
        radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_lg,
    ).set(
        body_background_fill="#FBFBFD", body_background_fill_dark="#000000",
        body_text_color="#1D1D1F", body_text_color_dark="#F5F5F7",
        body_text_color_subdued="#6E6E73", body_text_color_subdued_dark="#98989D",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#1C1C1E",
        background_fill_secondary="#F5F5F7", background_fill_secondary_dark="#2C2C2E",
        border_color_primary="rgba(0,0,0,0.08)", border_color_primary_dark="rgba(255,255,255,0.10)",
        block_background_fill="transparent", block_background_fill_dark="transparent",
        block_border_width="0px", block_shadow="none", block_padding="0px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_color="#6E6E73", block_label_text_color_dark="#98989D",
        block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
        block_title_text_color="#6E6E73", block_title_text_color_dark="#98989D",
        block_title_text_weight="500", block_title_text_size="13px", block_title_padding="0px",
        input_background_fill="#F5F5F7", input_background_fill_dark="#2C2C2E",
        input_border_color="transparent", input_border_color_dark="transparent",
        input_shadow="none", input_shadow_focus="0 0 0 4px rgba(0,113,227,0.18)",
        input_border_color_focus="#0071E3", input_border_color_focus_dark="#2997FF",
        button_primary_background_fill="#1D1D1F", button_primary_background_fill_dark="#F5F5F7",
        button_primary_background_fill_hover="#000000", button_primary_background_fill_hover_dark="#FFFFFF",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#000000",
        button_secondary_background_fill="rgba(0,0,0,0.05)", button_secondary_background_fill_dark="rgba(255,255,255,0.10)",
        button_secondary_background_fill_hover="rgba(0,0,0,0.08)",
        button_secondary_background_fill_hover_dark="rgba(255,255,255,0.14)",
        button_secondary_text_color="#1D1D1F", button_secondary_text_color_dark="#F5F5F7",
        button_secondary_border_color="transparent", button_secondary_border_color_dark="transparent",
        color_accent="#7C5CFF", color_accent_soft="rgba(124,92,255,0.10)",
        loader_color="#7C5CFF", shadow_drop="none", shadow_drop_lg="none",
    )


def load_css() -> str:
    return (UI_DIR / "style.css").read_text(encoding="utf-8")


def load_js() -> str:
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    return (script.replace("__ST_STATES__", json.dumps(UI_STATES, ensure_ascii=False))
            .replace("__ST_ISLAND__", json.dumps(ISLAND, ensure_ascii=False)))


MARK_SVG = """<svg class="st-mark" viewBox="0 0 28 28" aria-hidden="true">
<defs><linearGradient id="st-mark-g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#FF8A3D"/><stop offset=".35" stop-color="#FF3D7F"/>
<stop offset=".7" stop-color="#8B5CF6"/><stop offset="1" stop-color="#2F80FF"/></linearGradient></defs>
<rect width="28" height="28" rx="8" fill="url(#st-mark-g)"/>
<rect x="6.5" y="11.5" width="2.4" height="5" rx="1.2" fill="#fff"/><rect x="10.7" y="8" width="2.4" height="12" rx="1.2" fill="#fff"/>
<rect x="14.9" y="10" width="2.4" height="8" rx="1.2" fill="#fff"/><rect x="19.1" y="7" width="2.4" height="14" rx="1.2" fill="#fff"/>
</svg>"""


def topbar_html(engine_label: str, demo_mode: bool) -> str:
    badge = (
        '<span class="st-pill st-pill-demo" title="不运行模型，只返回内置示例的数据集参考文本">演示模式</span>'
        if demo_mode else
        f'<span class="st-pill" title="模型已在服务器本地加载"><span class="st-pill-dot"></span>{escape(engine_label)}</span>'
    )
    return f"""<header class="st-topbar">
  <a class="st-brand" href="./" aria-label="{PRODUCT_NAME}">{MARK_SVG}<span>{PRODUCT_NAME}</span></a>
  <div class="st-topbar-right">{badge}</div>
</header>"""


def hero_html(language_count: int, demo_mode: bool) -> str:
    notice = (
        '<p class="st-demo-note"><span>演示模式</span>未加载模型 · 内置示例返回 CoVoST 2 数据集参考文本，并非模型输出</p>'
        if demo_mode else ""
    )
    return f"""<section class="st-hero">
  <h1>开口说话，<span class="st-gradient-text">即刻跨越语言。</span></h1>
  <p class="st-lede">自动识别说话语言，原文与译文同时呈现。支持 {language_count} 种语言。</p>
  {notice}
</section>"""


STAGE_HTML = f"""<div class="st-stage-visual">
  <div class="st-island" id="st-island" role="status" aria-live="polite">
    <span class="st-island-inner" id="st-island-inner">
      <span class="st-island-icon" aria-hidden="true"></span>
      <span class="st-island-text" id="st-island-text">{ISLAND["idle"]}</span>
      <span class="st-island-meta" id="st-island-meta"></span>
    </span>
  </div>
  <div class="st-wave" aria-hidden="true"><canvas id="st-wave-canvas"></canvas></div>
</div>"""


def hint_html(max_seconds: float) -> str:
    return f"""<p class="st-hint">单段最长 {max_seconds:g} 秒 · 麦克风录音需 HTTPS 与浏览器授权 ·
<kbd>⌘</kbd><kbd>↵</kbd> 提交</p>"""


def section_head_html(title: str, note: str = "") -> str:
    note_html = f'<span class="st-section-note">{escape(note)}</span>' if note else ""
    return f'<div class="st-section-head"><h2>{escape(title)}</h2>{note_html}</div>'


INTERPRET_HTML = """<section class="st-soon">
  <span class="st-soon-badge">即将推出</span>
  <h1><span class="st-gradient-text">同声传译</span></h1>
  <p class="st-lede">边说边译。说话的同时持续输出双语字幕，适合会议、访谈与跨语言交流。</p>
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
