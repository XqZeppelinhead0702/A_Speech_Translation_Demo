() => {
  // Runs once on page load (Gradio `js=`). The visual state is derived from the status
  // box that app.py's event handlers already write, so no business logic lives here.
  const STATES = __ST_STATES__;
  const CAPTIONS = __ST_CAPTIONS__;
  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let state = null;
  let stateSince = performance.now();
  let prevState = "idle";

  const $ = (selector) => document.querySelector(selector);

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
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.72;
      ctx.createMediaStreamSource(stream).connect(analyser);
      Object.assign(session, { stream, ctx, analyser, bins: new Uint8Array(analyser.frequencyBinCount) });
    } catch (error) {
      // Permission or device failure: the visualizer falls back to a neutral pulse.
    }
  }
  function stopMic() {
    if (!mic) return;
    mic.stream?.getTracks().forEach((t) => t.stop());
    mic.ctx?.close();
    mic = null;
  }

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
    const caption = $("#st-viz-caption");
    if (caption) caption.textContent = CAPTIONS[next];
    if (next === "recording") startMic(); else stopMic();
    // On narrow screens the result sits below the fold; bring it into view once.
    if (next === "done" && prevState === "processing" && window.innerWidth < 900) {
      $("#st-output")?.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
    }
  }

  function syncLabels() {
    // Show the chosen target language beside the translation heading.
    const target = $("#st-target input");
    const label = $("#st-translation [data-testid='block-info']");
    if (target && label && label.dataset.lang !== target.value) label.dataset.lang = target.value;
    const source = $("#st-source input");
    const swap = $("#st-swap");
    if (source && swap) swap.dataset.auto = String(source.value === "自动检测");
  }

  function tick() {
    const status = $("#st-status textarea, #st-status input");
    setState(deriveState(status ? status.value : ""));
    syncLabels();
  }
  setInterval(tick, 120);
  tick();

  // ---- Voice ribbon visualizer -----------------------------------------------
  const BARS = 48;
  const levels = new Float32Array(BARS).fill(0.04);
  const voiceprint = Array.from({ length: BARS }, (_, i) => {
    const x = i / (BARS - 1);
    const envelope = Math.sin(Math.PI * x) ** 0.8;
    return 0.12 + envelope * (0.35 + 0.35 * Math.abs(Math.sin(i * 1.9) * Math.cos(i * 0.47)));
  });

  function targetLevel(i, now) {
    const t = reducedMotion.matches ? 0 : now;
    const x = i / (BARS - 1);
    const age = now - stateSince;
    switch (state) {
      case "recording": {
        if (mic?.analyser) {
          // Mirror the low/mid spectrum around the centre so speech reads symmetric.
          const offset = Math.abs(x - 0.5) * 2;
          const bin = Math.min(mic.bins.length - 1, Math.floor(2 + offset * 46));
          return 0.05 + (mic.bins[bin] / 255) ** 1.2 * (1.05 - offset * 0.35);
        }
        return 0.12 + 0.08 * Math.sin(t / 260 + i * 0.5);
      }
      case "ready":
        return voiceprint[i] * (0.92 + 0.08 * Math.sin(t / 900 + i * 0.3));
      case "processing": {
        const phase = ((age / 1500) % 1) * 1.5 - 0.25;
        const packet = Math.exp(-(((x - phase) * 5.5) ** 2));
        return 0.07 + packet * (0.55 + 0.25 * Math.sin(t / 110 + i * 1.3)) + 0.03 * Math.sin(t / 400 + i);
      }
      case "done": {
        // Settle from the processing burst into a quiet copy of the voiceprint.
        const settle = Math.max(0, 1 - age / 900);
        return 0.04 + voiceprint[i] * (0.3 + 0.4 * settle) + 0.015 * Math.sin(t / 1400 + i * 0.4);
      }
      case "error":
        return 0.04 + (age < 500 ? 0.12 * Math.abs(Math.sin(age / 40 + i)) : 0);
      default:
        return 0.05 + 0.035 * (1 + Math.sin(t / 1100 - i * 0.28));
    }
  }

  let palette = null;
  let paletteAt = 0;
  function readPalette(canvas, now) {
    if (palette && now - paletteAt < 1000) return palette;
    const style = getComputedStyle(canvas);
    palette = {
      ink: style.getPropertyValue("--st-viz-ink").trim() || "#8C8993",
      accent: style.getPropertyValue("--st-accent").trim() || "#F2542D",
      danger: style.getPropertyValue("--st-danger").trim() || "#D93A3A",
    };
    paletteAt = now;
    return palette;
  }

  function draw(now) {
    const canvas = $("#st-viz-canvas");
    if (canvas && canvas.offsetParent !== null) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(canvas.clientWidth * dpr);
      const h = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      if (mic?.analyser) mic.analyser.getByteFrequencyData(mic.bins);
      const ctx = canvas.getContext("2d");
      const colors = readPalette(canvas, now);
      ctx.clearRect(0, 0, w, h);
      const gap = w / BARS;
      const barWidth = Math.max(2 * dpr, gap * 0.42);
      const ease = reducedMotion.matches ? 1 : state === "recording" ? 0.35 : 0.14;
      const live = state === "recording" || state === "processing";
      let gradient = null;
      if (state === "processing") {
        gradient = ctx.createLinearGradient(0, 0, w, 0);
        gradient.addColorStop(0, colors.ink);
        gradient.addColorStop(1, colors.accent);
      }
      for (let i = 0; i < BARS; i++) {
        levels[i] += (Math.min(1, targetLevel(i, now)) - levels[i]) * ease;
        const barHeight = Math.max(barWidth, levels[i] * h * 0.92);
        const x = gap * i + (gap - barWidth) / 2;
        const y = (h - barHeight) / 2;
        ctx.fillStyle = state === "error" ? colors.danger : state === "recording" ? colors.accent : gradient || colors.ink;
        ctx.globalAlpha = live || state === "ready" ? 1 : 0.55;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2) : ctx.rect(x, y, barWidth, barHeight);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
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
