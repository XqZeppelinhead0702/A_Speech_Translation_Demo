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
# app.js derives the visual state (voice line, status label) from them.
UI_STATES = [
    ("正在录音", "recording"),
    ("音频已就绪", "ready"),
    ("排队或处理中", "processing"),
    ("完成", "done"),
    ("处理未完成", "error"),
]
# Status label copy per state; app.js appends live timers.
STATUS_COPY = {
    "idle": "等待音频",
    "recording": "正在聆听",
    "ready": "音频已就绪",
    "processing": "排队中",
    "done": "完成",
    "error": "未能完成",
}
# Processing stages the translator reports ("排队或处理中 · 正在转写") -> label copy.
STAGE_COPY = {
    "正在识别语言": "正在识别语言",
    "正在转写": "正在转写原文",
    "正在翻译": "正在翻译",
}

FONTS_HEAD = """
<meta name="theme-color" content="#FAFAF9" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0B0B0B" media="(prefers-color-scheme: dark)">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap">
"""

# Native first: SF Pro + PingFang on Apple devices, Inter + YaHei / Noto elsewhere.
SANS = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "PingFang SC",
        "HarmonyOS Sans SC", "Microsoft YaHei", "Noto Sans SC", "system-ui", "sans-serif"]
MONO = ["SF Mono", "ui-monospace", "SFMono-Regular", "JetBrains Mono", "Menlo", "monospace"]

INK, INK_DARK = "#171717", "#EDEDED"


def build_theme() -> gr.themes.Base:
    """Base theme mapped onto the design tokens; style.css refines the details."""
    return gr.themes.Base(
        primary_hue="neutral", neutral_hue="neutral", font=SANS, font_mono=MONO,
        radius_size=gr.themes.sizes.radius_md, spacing_size=gr.themes.sizes.spacing_lg,
    ).set(
        body_background_fill="#FAFAF9", body_background_fill_dark="#0B0B0B",
        body_text_color=INK, body_text_color_dark=INK_DARK,
        body_text_color_subdued="#737373", body_text_color_subdued_dark="#8F8F8F",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#121212",
        background_fill_secondary="#F4F4F3", background_fill_secondary_dark="#1A1A1A",
        border_color_primary="rgba(0,0,0,0.08)", border_color_primary_dark="rgba(255,255,255,0.08)",
        block_background_fill="transparent", block_background_fill_dark="transparent",
        block_border_width="0px", block_shadow="none", block_padding="0px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_color="#737373", block_label_text_color_dark="#8F8F8F",
        block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
        block_title_text_color="#737373", block_title_text_color_dark="#8F8F8F",
        block_title_text_weight="500", block_title_text_size="12px", block_title_padding="0px",
        input_background_fill="transparent", input_background_fill_dark="transparent",
        input_border_color="transparent", input_border_color_dark="transparent",
        input_shadow="none", input_shadow_focus="none",
        input_border_color_focus="transparent", input_border_color_focus_dark="transparent",
        button_primary_background_fill=INK, button_primary_background_fill_dark=INK_DARK,
        button_primary_background_fill_hover="#000000", button_primary_background_fill_hover_dark="#FFFFFF",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0B0B0B",
        button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="rgba(0,0,0,0.04)",
        button_secondary_background_fill_hover_dark="rgba(255,255,255,0.06)",
        button_secondary_text_color=INK, button_secondary_text_color_dark=INK_DARK,
        button_secondary_border_color="rgba(0,0,0,0.10)", button_secondary_border_color_dark="rgba(255,255,255,0.12)",
        color_accent=INK, color_accent_soft="rgba(0,0,0,0.05)", color_accent_soft_dark="rgba(255,255,255,0.08)",
        loader_color=INK, loader_color_dark=INK_DARK, shadow_drop="none", shadow_drop_lg="none",
        checkbox_label_background_fill_selected=INK,
    )


def load_css() -> str:
    return (UI_DIR / "style.css").read_text(encoding="utf-8")


def load_js() -> str:
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    return (script.replace("__ST_STATES__", json.dumps(UI_STATES, ensure_ascii=False))
            .replace("__ST_COPY__", json.dumps(STATUS_COPY, ensure_ascii=False))
            .replace("__ST_STAGES__", json.dumps(STAGE_COPY, ensure_ascii=False)))


# Monochrome mark: one continuous line that leaves the baseline as a voice and returns.
MARK_SVG = """<svg class="st-mark" viewBox="0 0 20 20" aria-hidden="true">
<rect width="20" height="20" rx="5.5" fill="currentColor"/>
<path d="M4 10h2.2c.9 0 1.2-4.2 2.2-4.2S9.6 14.2 10.6 14.2 11.8 8 12.8 8s1.1 2 2 2H16" fill="none"
 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def topbar_html(engine_label: str, demo_mode: bool) -> str:
    badge = (
        '<span class="st-tag" title="不运行模型，只返回内置示例的数据集参考文本">演示模式</span>'
        if demo_mode else
        f'<span class="st-engine" title="模型已在服务器本地加载"><i aria-hidden="true"></i>{escape(engine_label)}</span>'
    )
    return f"""<header class="st-topbar">
  <a class="st-brand" href="./" aria-label="{PRODUCT_NAME}">{MARK_SVG}<span>{PRODUCT_NAME}</span></a>
  <div class="st-topbar-right">{badge}</div>
</header>"""


def hero_html(language_count: int, demo_mode: bool) -> str:
    notice = (
        '<p class="st-demo-note"><span class="st-tag">演示模式</span>未加载模型。内置示例返回 CoVoST 2 数据集的参考文本，并非模型输出。</p>'
        if demo_mode else ""
    )
    return f"""<section class="st-hero">
  <h1>开口说话，<span>即刻跨越语言。</span></h1>
  <p class="st-lede">自动识别说话语言，原文与译文同时呈现。支持 {language_count} 种语言。</p>
  {notice}
</section>"""


STAGE_HTML = f"""<div class="st-stage-visual">
  <div class="st-status-label" id="st-label" role="status" aria-live="polite">
    <span class="st-label-icon" aria-hidden="true"></span>
    <span class="st-label-text" id="st-label-text">{STATUS_COPY["idle"]}</span>
    <span class="st-label-meta" id="st-label-meta"></span>
  </div>
  <div class="st-wave" aria-hidden="true"><canvas id="st-wave-canvas"></canvas></div>
</div>"""


def hint_html(max_seconds: float) -> str:
    return f"""<p class="st-hint">单段最长 {max_seconds:g} 秒 · 麦克风录音需 HTTPS 与浏览器授权<span class="st-hint-keys"> ·
<kbd>⌘</kbd><kbd>↵</kbd> 提交</span></p>"""


def section_head_html(title: str, note: str = "") -> str:
    note_html = f'<span class="st-section-note">{escape(note)}</span>' if note else ""
    return f'<div class="st-section-head"><h2>{escape(title)}</h2>{note_html}</div>'


INTERPRET_HTML = """<section class="st-soon">
  <span class="st-tag">即将推出</span>
  <h1>同声传译</h1>
  <p class="st-lede">边说边译。说话的同时持续输出双语字幕，适合会议、访谈与跨语言交流。</p>
  <ol class="st-soon-grid">
    <li><span class="st-soon-num">01</span><h3>流式识别</h3><p>语音分段送入模型，字幕随说话实时出现。</p></li>
    <li><span class="st-soon-num">02</span><h3>双语对照</h3><p>原文与译文逐句对齐，回看与复制都很方便。</p></li>
    <li><span class="st-soon-num">03</span><h3>共享语言与模型</h3><p>沿用当前 21 种语言和本地部署方式。</p></li>
  </ol>
  <p class="st-soon-foot">该功能仍在规划中，当前版本不提供实时传译。</p>
</section>"""


def footer_html(engine_label: str) -> str:
    return f"""<footer class="st-footer">
  <span>{escape(engine_label)}</span><span aria-hidden="true">·</span>
  <span>临时音频约 1 小时后自动清理</span><span aria-hidden="true">·</span><span>结果可能有误，重要内容请人工核对</span>
</footer>"""
