() => {
  // Runs once on page load (Gradio `js=`). The visual state is derived from the status
  // box that app.py's event handlers already write, so no business logic lives here.
  const STATES = __ST_STATES__;
  const ISLAND = __ST_ISLAND__;
  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const $ = (selector) => document.querySelector(selector);

  let state = null;
  let prevState = "idle";
  let stateSince = performance.now();
  let statusText = "";

  // ---- Live microphone level, only while Gradio is recording -------------------
  let mic = null;
  async function startMic() {
    if (mic || !navigator.mediaDevices?.getUserMedia) return;
    const session = {};
    mic = session;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (mic !== session) { stream.getTracks().forEach((t) => t.stop()); return; }
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.6;
      ctx.createMediaStreamSource(stream).connect(analyser);
      Object.assign(session, { stream, ctx, analyser, samples: new Float32Array(analyser.fftSize) });
    } catch (error) {
      // Permission or device failure: the wave keeps a neutral listening pulse.
    }
  }
  function stopMic() {
    if (!mic) return;
    mic.stream?.getTracks().forEach((t) => t.stop());
    mic.ctx?.close();
    mic = null;
  }
  function micLevel() {
    if (!mic?.analyser) return null;
    mic.analyser.getFloatTimeDomainData(mic.samples);
    let sum = 0;
    for (const v of mic.samples) sum += v * v;
    const rms = Math.sqrt(sum / mic.samples.length);
    // Perceptual curve: quiet speech still moves the wave, shouting does not clip it.
    return Math.min(1, Math.pow(rms * 7, 0.7));
  }

  // ---- State ------------------------------------------------------------------
  function deriveState(text) {
    for (const [prefix, name] of STATES) if (text.startsWith(prefix)) return name;
    return "idle";
  }

  function setState(next) {
    if (next === state) return;
    prevState = state || "idle";
    state = next;
    stateSince = performance.now();
    root.dataset.stState = next;
    if (next === "recording") startMic(); else stopMic();
    renderIsland(true);
    if (next === "done" && prevState === "processing") {
      revealResults();
      if (window.innerWidth < 900) {
        $("#st-output")?.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
      }
    }
  }

  // ---- Status island: one capsule that morphs between states -------------------
  function formatClock(ms) {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  function islandMeta(now) {
    const age = now - stateSince;
    if (state === "recording") return formatClock(age);
    if (state === "processing") return `${(age / 1000).toFixed(1)}s`;
    if (state === "done") {
      const match = statusText.match(/用时\s*([\d.]+)\s*秒/);
      return match ? `${match[1]}s` : "";
    }
    return "";
  }
  let islandKey = "";
  function renderIsland(stateChanged, now = performance.now()) {
    const island = $("#st-island");
    const inner = $("#st-island-inner");
    const text = $("#st-island-text");
    const meta = $("#st-island-meta");
    if (!island || !inner || !text || !meta) return;
    const nextMeta = islandMeta(now);
    if (stateChanged) {
      island.dataset.state = state;
      // Cross-fade the label so the capsule morphs instead of snapping.
      inner.classList.remove("st-island-swap");
      void inner.offsetWidth;
      inner.classList.add("st-island-swap");
      text.textContent = ISLAND[state] || "";
    }
    if (meta.textContent !== nextMeta) meta.textContent = nextMeta;
    const key = `${state}|${nextMeta.length}`;
    if (key !== islandKey) {
      islandKey = key;
      island.style.width = `${Math.ceil(inner.scrollWidth) + 28}px`;
    }
  }

  // ---- Result reveal: words resolve from blur, in reading order ----------------
  function tokenize(value) {
    // CJK by character, everything else by word (keeping the following spaces).
    return value.match(/[　-鿿가-힯＀-￯]|[^\s　-鿿가-힯＀-￯]+\s*|\s+/g) || [];
  }
  function reveal(selector, baseDelay) {
    const field = $(`${selector} textarea`);
    if (!field || !field.value || reducedMotion.matches) return;
    const host = field.parentElement;
    host.querySelector(".st-reveal")?.remove();
    const layer = document.createElement("div");
    layer.className = "st-reveal";
    layer.setAttribute("aria-hidden", "true");
    const tokens = tokenize(field.value);
    const step = Math.min(38, 900 / Math.max(tokens.length, 1));
    tokens.forEach((token, index) => {
      const span = document.createElement("span");
      span.textContent = token;
      span.style.animationDelay = `${baseDelay + index * step}ms`;
      layer.appendChild(span);
    });
    host.appendChild(layer);
    field.classList.add("st-revealing");
    const total = baseDelay + tokens.length * step + 700;
    setTimeout(() => { layer.remove(); field.classList.remove("st-revealing"); }, total);
  }
  function revealResults() {
    // Wait one frame so Gradio has written the new values into the textareas.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      reveal("#st-transcript", 0);
      reveal("#st-translation", 220);
    }));
  }

  function syncLabels() {
    const target = $("#st-target input");
    const label = $("#st-translation [data-testid='block-info']");
    if (target && label && label.dataset.lang !== target.value) label.dataset.lang = target.value;
    const source = $("#st-source input");
    const swap = $("#st-swap");
    if (source && swap) swap.dataset.auto = String(source.value === "自动检测");
    // The detected-language line only appears once there is something to say.
    const detected = $("#st-lang-result");
    const detectedText = $("#st-lang-result textarea");
    if (detected && detectedText) detected.classList.toggle("st-empty", detectedText.value === "等待检测" || !detectedText.value);
  }

  function tick() {
    const status = $("#st-status textarea, #st-status input");
    statusText = status ? status.value : "";
    setState(deriveState(statusText));
    syncLabels();
    renderIsland(false);
  }
  setInterval(tick, 100);
  tick();

  // ---- Voice wave: layered, attenuated sine ribbons (Siri-like) ----------------
  const CURVES = [
    { rgb: [255, 138, 61], amp: 1.0, freq: 1.0, speed: 0.9, offset: 0.0 },
    { rgb: [255, 61, 127], amp: 0.82, freq: 1.4, speed: 1.2, offset: 1.3 },
    { rgb: [139, 92, 246], amp: 0.7, freq: 1.8, speed: 0.75, offset: 2.6 },
    { rgb: [47, 128, 255], amp: 0.58, freq: 2.3, speed: 1.4, offset: 3.9 },
  ];
  const ERROR_RGB = [255, 69, 58];
  const phases = CURVES.map((c) => c.offset);
  let level = 0.05;
  let last = performance.now();

  function targetLevel(now) {
    const age = now - stateSince;
    const breathe = reducedMotion.matches ? 0 : Math.sin(now / 1300);
    switch (state) {
      case "recording": {
        const live = micLevel();
        return live === null ? 0.18 + 0.06 * breathe : 0.06 + live * 0.94;
      }
      case "ready": return 0.2 + 0.04 * breathe;
      case "processing": return 0.36 + 0.12 * (reducedMotion.matches ? 0 : Math.sin(now / 520));
      case "done": return 0.05 + 0.3 * Math.max(0, 1 - age / 1100);
      case "error": return 0.04;
      default: return 0.055 + 0.02 * breathe;
    }
  }
  function speedFor() {
    return { recording: 1.8, processing: 2.6, ready: 0.7, done: 0.9, error: 0.4 }[state] || 0.55;
  }

  let palette = null;
  let paletteAt = 0;
  function readPalette(el, now) {
    if (palette && now - paletteAt < 800) return palette;
    const style = getComputedStyle(el);
    palette = {
      line: style.getPropertyValue("--st-wave-line").trim() || "rgba(0,0,0,.18)",
      additive: style.getPropertyValue("--st-wave-blend").trim() === "lighter",
    };
    paletteAt = now;
    return palette;
  }

  function draw(now) {
    const canvas = $("#st-wave-canvas");
    const dt = Math.min(64, now - last);
    last = now;
    if (canvas && canvas.offsetParent !== null) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(canvas.clientWidth * dpr);
      const h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      const target = targetLevel(now);
      // Fast attack, slow release: the wave jumps with a syllable and exhales after it.
      const k = reducedMotion.matches ? 1 : target > level ? 0.32 : 0.07;
      level += (target - level) * k;
      const stage = canvas.closest(".st-stage-visual");
      if (stage) stage.style.setProperty("--st-level", level.toFixed(3));

      const ctx = canvas.getContext("2d");
      const colors = readPalette(canvas, now);
      ctx.clearRect(0, 0, w, h);
      const mid = h / 2;
      const age = now - stateSince;
      const shake = state === "error" && !reducedMotion.matches ? Math.sin(age / 28) * 7 * dpr * Math.exp(-age / 260) : 0;

      // Hairline horizon, always present so silence still reads as "ready to listen".
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.strokeStyle = colors.line;
      ctx.lineWidth = dpr;
      ctx.beginPath();
      ctx.moveTo(w * 0.06, mid);
      ctx.lineTo(w * 0.94, mid);
      ctx.stroke();

      ctx.globalCompositeOperation = colors.additive ? "lighter" : "source-over";
      const speed = speedFor();
      CURVES.forEach((curve, i) => {
        if (!reducedMotion.matches) phases[i] += (dt / 1000) * curve.speed * speed * 2.2;
        const pulse = reducedMotion.matches ? 1 : 0.62 + 0.38 * Math.sin(now / (900 + i * 210) + curve.offset);
        const amplitude = mid * 0.92 * level * curve.amp * pulse;
        const rgb = state === "error" ? ERROR_RGB : curve.rgb;
        const gradient = ctx.createLinearGradient(0, mid - amplitude, 0, mid + amplitude);
        const alpha = colors.additive ? 0.55 : 0.42;
        gradient.addColorStop(0, `rgba(${rgb},0)`);
        gradient.addColorStop(0.5, `rgba(${rgb},${alpha})`);
        gradient.addColorStop(1, `rgba(${rgb},0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        const steps = 120;
        for (let s = 0; s <= steps; s++) {
          const nx = (s / steps) * 4 - 2;
          const attenuation = Math.pow(4 / (4 + Math.pow(nx, 4)), 4);
          const y = mid - amplitude * attenuation * Math.sin(curve.freq * nx * Math.PI - phases[i]);
          const x = (s / steps) * w * 0.88 + w * 0.06 + shake;
          s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        for (let s = steps; s >= 0; s--) {
          const nx = (s / steps) * 4 - 2;
          const attenuation = Math.pow(4 / (4 + Math.pow(nx, 4)), 4);
          const y = mid + amplitude * attenuation * Math.sin(curve.freq * nx * Math.PI - phases[i]);
          ctx.lineTo((s / steps) * w * 0.88 + w * 0.06 + shake, y);
        }
        ctx.closePath();
        ctx.fill();
      });
      ctx.globalCompositeOperation = "source-over";
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);

  // ---- Keyboard: Ctrl/⌘ + Enter submits when the button is enabled ------------
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      const button = $("#st-submit");
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    }
  });
}
