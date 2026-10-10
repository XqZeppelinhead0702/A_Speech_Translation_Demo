() => {
  // Runs once on page load (Gradio `js=`). The visual state is derived from the status
  // box that app.py's event handlers already write, so no business logic lives here.
  const STATES = __ST_STATES__;
  const COPY = __ST_COPY__;
  const STAGES = __ST_STAGES__;
  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const $ = (selector) => document.querySelector(selector);
  const still = () => reducedMotion.matches;
  // Palette preview: ?palette=sage or ?palette=dusk swaps the colour tokens (style.css).
  const palette = new URLSearchParams(location.search).get("palette");
  if (palette && /^[a-z]+$/.test(palette)) root.dataset.stPalette = palette;

  let state = null;
  let prevState = "idle";
  let stateSince = performance.now();
  let statusText = "";

  // ---- Live microphone spectrum, only while Gradio is recording ----------------
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
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.5;
      ctx.createMediaStreamSource(stream).connect(analyser);
      Object.assign(session, {
        stream, ctx, analyser,
        wave: new Float32Array(analyser.fftSize),
        spectrum: new Uint8Array(analyser.frequencyBinCount),
      });
    } catch (error) {
      // Permission or device failure: the line keeps a neutral listening breath.
    }
  }
  function stopMic() {
    if (!mic) return;
    mic.stream?.getTracks().forEach((t) => t.stop());
    mic.ctx?.close();
    mic = null;
  }
  // Loudness plus the energy of a few voice bands (80 Hz – 4 kHz, log spaced), so the
  // line's shape follows the timbre of what is said, not a canned animation.
  const BANDS = 6;
  function micFrame() {
    if (!mic?.analyser) return null;
    const { analyser, wave, spectrum, ctx } = mic;
    analyser.getFloatTimeDomainData(wave);
    let sum = 0;
    for (const v of wave) sum += v * v;
    const level = Math.min(1, Math.pow(Math.sqrt(sum / wave.length) * 6, 0.75));
    analyser.getByteFrequencyData(spectrum);
    const hz = ctx.sampleRate / analyser.fftSize;
    const bands = [];
    for (let b = 0; b < BANDS; b++) {
      const lo = Math.floor((80 * Math.pow(50, b / BANDS)) / hz);
      const hi = Math.max(lo + 1, Math.floor((80 * Math.pow(50, (b + 1) / BANDS)) / hz));
      let peak = 0;
      for (let i = lo; i < hi && i < spectrum.length; i++) peak = Math.max(peak, spectrum[i]);
      bands.push(Math.max(0, (peak - 90) / 140));
    }
    return { level, bands };
  }

  // ---- Recorded clip: decoded once into a loudness profile --------------------
  let clip = { src: "", peaks: null, duration: 0 };
  async function loadClip(src) {
    if (!src || clip.src === src) return;
    clip = { src, peaks: null, duration: 0 };
    try {
      const data = await (await fetch(src)).arrayBuffer();
      const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      const buffer = await new Offline(1, 1, 44100).decodeAudioData(data);
      const samples = buffer.getChannelData(0);
      const buckets = 240;
      const size = Math.max(1, Math.floor(samples.length / buckets));
      const peaks = new Float32Array(buckets);
      let max = 1e-6;
      for (let b = 0; b < buckets; b++) {
        let sum = 0;
        for (let i = b * size, end = Math.min(samples.length, i + size); i < end; i++) sum += samples[i] * samples[i];
        peaks[b] = Math.sqrt(sum / size);
        max = Math.max(max, peaks[b]);
      }
      for (let b = 0; b < buckets; b++) peaks[b] = Math.pow(peaks[b] / max, 0.8);
      if (clip.src === src) Object.assign(clip, { peaks, duration: buffer.duration, since: performance.now() });
    } catch (error) {
      // Undecodable here (codec, network): the stage falls back to the plain line.
    }
  }
  function playerEl() { return $("#st-audio audio"); }

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
    if (next === "idle" || next === "recording") clip = { src: "", peaks: null, duration: 0 };
    if (next === "done" && window.innerWidth < 900) {
      setTimeout(() => $("#st-output")?.scrollIntoView({ behavior: still() ? "auto" : "smooth", block: "start" }), 120);
    }
  }

  // ---- Status label: the stage actually running, plus a live timer -------------
  let processingSince = 0;
  function labelCopy() {
    if (state === "processing") {
      const stage = statusText.split("·")[1]?.trim() || "";
      return STAGES[stage] || COPY.processing;
    }
    return COPY[state] || "";
  }
  function labelMeta(now) {
    if (state === "recording") {
      const s = Math.floor((now - stateSince) / 1000);
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }
    if (state === "processing") return `${((now - processingSince) / 1000).toFixed(1)}s`;
    if (state === "done") {
      const match = statusText.match(/用时\s*([\d.]+)\s*秒/);
      return match ? `${match[1]}s` : "";
    }
    if (state === "ready" && clip.duration) return `${clip.duration.toFixed(1)}s`;
    return "";
  }
  function renderLabel(now) {
    const label = $("#st-label");
    const text = $("#st-label-text");
    const meta = $("#st-label-meta");
    if (!label || !text || !meta) return;
    if (label.dataset.state !== state) {
      label.dataset.state = state;
      if (state === "processing") processingSince = now;
    }
    const copy = labelCopy();
    if (text.textContent !== copy) {
      text.textContent = copy;
      text.classList.remove("st-swap-in");
      void text.offsetWidth;
      text.classList.add("st-swap-in");
    }
    const nextMeta = labelMeta(now);
    if (meta.textContent !== nextMeta) meta.textContent = nextMeta;
  }

  // ---- Results: each field reveals the moment its text arrives ----------------
  function tokenize(value) {
    // CJK by character, everything else by word (keeping the following spaces).
    return value.match(/[　-鿿가-힯＀-￯]|[^\s　-鿿가-힯＀-￯]+\s*|\s+/g) || [];
  }
  function reveal(field) {
    if (still()) return;
    const host = field.parentElement;
    host.querySelector(".st-reveal")?.remove();
    const layer = document.createElement("div");
    layer.className = "st-reveal";
    layer.setAttribute("aria-hidden", "true");
    const tokens = tokenize(field.value);
    const step = Math.min(26, 640 / Math.max(tokens.length, 1));
    tokens.forEach((token, index) => {
      if (token === "\n") { layer.appendChild(document.createElement("br")); return; }
      const span = document.createElement("span");
      span.textContent = token;
      span.style.animationDelay = `${index * step}ms`;
      layer.appendChild(span);
    });
    host.appendChild(layer);
    field.classList.add("st-revealing");
    setTimeout(() => { layer.remove(); field.classList.remove("st-revealing"); }, tokens.length * step + 600);
  }
  const seen = {};
  function watchResults() {
    for (const id of ["st-transcript", "st-translation"]) {
      const field = $(`#${id} textarea`);
      if (!field) continue;
      const value = field.value;
      if (value && !seen[id] && (state === "processing" || state === "done")) reveal(field);
      seen[id] = value;
    }
  }

  function syncLabels() {
    const target = $("#st-target input");
    const label = $("#st-translation [data-testid='block-info']");
    const lang = target ? target.value.split(" · ")[0] : "";
    if (label && label.dataset.lang !== lang) label.dataset.lang = lang;
    const source = $("#st-source input");
    const swap = $("#st-swap");
    if (source && swap) swap.dataset.auto = String(source.value === "自动检测");
    // The detected-language line only appears once there is something to say.
    const detected = $("#st-lang-result");
    const detectedText = $("#st-lang-result textarea");
    if (detected && detectedText) detected.classList.toggle("st-empty", detectedText.value === "等待检测" || !detectedText.value);
    if (state === "ready" || state === "processing" || state === "done" || state === "error") loadClip(playerEl()?.src);
  }

  function tick() {
    const status = $("#st-status textarea, #st-status input");
    statusText = status ? status.value : "";
    setState(deriveState(statusText));
    syncLabels();
    watchResults();
  }

  // ---- Voice line --------------------------------------------------------------
  // One accent, two strokes. Silence is a still hairline. Speech bends it, shaped by the
  // live voice bands. A finished recording settles into its own loudness profile, the
  // system reads across it while processing, and playback fills it in.
  const spring = (current, target, up, down) => current + (target - current) * (target > current ? up : down);
  let level = 0;
  let bands = new Array(BANDS).fill(0);
  let phase = 0;
  let profileMix = 0;
  let last = performance.now();
  let lastKey = "";
  let colors = null;
  let colorsAt = 0;
  let probe = null;
  function readColors(el, now) {
    if (colors && now - colorsAt < 600) return colors;
    // Resolve the tokens through a probe so the canvas gets plain computed colours.
    if (!probe || !probe.isConnected) { probe = document.createElement("i"); el.parentElement.appendChild(probe); }
    const get = (name) => { probe.style.color = `var(${name})`; return getComputedStyle(probe).color; };
    colors = { accent: get("--st-accent"), ink: get("--st-accent-deep"), ink4: get("--st-ink-4") };
    colorsAt = now;
    return colors;
  }
  const windowAt = (u) => Math.pow(Math.sin(Math.PI * u), 2);

  function drawLine(ctx, w, h, dpr, alpha, color, amp) {
    const mid = h / 2;
    const left = w * 0.04;
    const span = w * 0.92;
    const strokes = [{ scale: 0.62, shift: 1.9, width: 1, opacity: 0.32 }, { scale: 1, shift: 0, width: 1.6, opacity: 1, fill: true }];
    for (const stroke of strokes) {
      ctx.globalAlpha = alpha * stroke.opacity;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = stroke.width * dpr;
      ctx.beginPath();
      const steps = 160;
      for (let s = 0; s <= steps; s++) {
        const u = s / steps;
        let y = 0;
        for (let b = 0; b < BANDS; b++) {
          y += (0.25 + bands[b]) * Math.sin(2 * Math.PI * (0.8 + b * 0.62) * u + phase * (1 + b * 0.17) + stroke.shift * (b + 1));
        }
        y = (y / BANDS) * 2.2 * amp * stroke.scale * windowAt(u);
        s === 0 ? ctx.moveTo(left, mid - y) : ctx.lineTo(left + u * span, mid - y);
      }
      ctx.stroke();
      if (stroke.fill) {
        // A faint wash between the line and its baseline gives the voice some body.
        ctx.lineTo(left + span, mid);
        ctx.lineTo(left, mid);
        ctx.closePath();
        ctx.globalAlpha = alpha * 0.08;
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawProfile(ctx, w, h, dpr, mix, c, now, shake) {
    const peaks = clip.peaks;
    const mid = h / 2;
    const left = w * 0.04;
    const span = w * 0.92;
    const gap = 3.5 * dpr;
    const count = Math.max(24, Math.floor(span / gap));
    const player = playerEl();
    const played = player && player.duration && (player.currentTime > 0) ? player.currentTime / player.duration : 0;
    // Processing: a soft reading head sweeps across, inking the bars it passes.
    let head = -1;
    if (state === "processing" && !still()) {
      const t = ((now - stateSince) % 1900) / 1900;
      head = -0.15 + 1.3 * (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    }
    ctx.lineCap = "round";
    ctx.lineWidth = 1.5 * dpr;
    for (let i = 0; i < count; i++) {
      const u = i / (count - 1);
      const peak = peaks[Math.min(peaks.length - 1, Math.floor(u * peaks.length))];
      const height = Math.max(0.75 * dpr, peak * mid * 0.72 * mix);
      // Resting bars sit at a soft accent; played and freshly read bars come to full.
      let ink = state === "processing" && !still() ? 0.22 : state === "error" ? 0.3 : (document.body.classList.contains("dark") ? 0.6 : 0.45);
      if (played && u <= played) ink = 1;
      if (head > -1) ink = Math.max(ink, Math.exp(-Math.pow((u - head) / 0.07, 2)));
      const x = left + u * span + shake;
      ctx.strokeStyle = ink > 0.9 && played && u <= played ? c.ink : c.accent;
      ctx.globalAlpha = ink;
      ctx.beginPath();
      ctx.moveTo(x, mid - height);
      ctx.lineTo(x, mid + height);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function draw(now) {
    requestAnimationFrame(draw);
    const canvas = $("#st-wave-canvas");
    const dt = Math.min(64, now - last);
    last = now;
    if (!canvas || canvas.offsetParent === null) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    const resized = canvas.width !== w || canvas.height !== h;
    if (resized) { canvas.width = w; canvas.height = h; }

    // Targets for this frame.
    let targetLevel = 0;
    let targetBands = new Array(BANDS).fill(0);
    if (state === "recording") {
      const frame = micFrame();
      if (frame) { targetLevel = frame.level; targetBands = frame.bands; }
      else targetLevel = 0.12;
    }
    // Fast attack, slow release: the line jumps with a syllable and exhales after it.
    level = still() ? targetLevel : spring(level, targetLevel, 0.3, 0.06);
    bands = bands.map((value, b) => (still() ? targetBands[b] : spring(value, targetBands[b], 0.35, 0.08)));
    if (!still() && state === "recording") phase += (dt / 1000) * (0.9 + level * 2.2);
    const wantProfile = Boolean(clip.peaks) && state !== "recording" && state !== "idle";
    profileMix = still() ? Number(wantProfile) : spring(profileMix, Number(wantProfile), 0.12, 0.2);

    const player = playerEl();
    const playing = player && !player.paused;
    const transitioning = now - stateSince < 900 || Math.abs(profileMix - Number(wantProfile)) > 0.002 || level > 0.002;
    const animated = state === "recording" || (state === "processing" && !still()) || playing || transitioning;
    const key = `${state}|${clip.src}|${Boolean(clip.peaks)}|${w}x${h}|${colors?.ink}`;
    if (!animated && !resized && key === lastKey) return;
    lastKey = key;

    const ctx = canvas.getContext("2d");
    const c = readColors(canvas, now);
    ctx.clearRect(0, 0, w, h);
    const mid = h / 2;
    const age = now - stateSince;
    const shake = state === "error" && !still() ? Math.sin(age / 32) * 4 * dpr * Math.exp(-age / 180) : 0;

    // The line: still hairline at rest, bending with the voice while listening.
    const lineAlpha = 1 - profileMix;
    if (lineAlpha > 0.01) {
      const breath = state === "recording" && !still() ? (1.2 + 0.8 * Math.sin(now / 900)) * dpr : 0;
      const amp = breath + level * mid * 1.05;
      if (amp < 0.3 * dpr) {
        ctx.globalAlpha = lineAlpha * (state === "idle" ? 0.4 : 0.55);
        // No clip profile to read across while processing: a short segment travels the line.
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = dpr;
        ctx.beginPath();
        ctx.moveTo(w * 0.04 + shake, mid);
        ctx.lineTo(w * 0.96 + shake, mid);
        ctx.stroke();
        if (state === "processing" && !still()) {
          const t = (age % 1600) / 1600;
          const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
          const x = w * 0.04 + w * 0.92 * e;
          const gradient = ctx.createLinearGradient(x - w * 0.12, 0, x + w * 0.12, 0);
          gradient.addColorStop(0, "rgba(0,0,0,0)");
          gradient.addColorStop(0.5, c.accent);
          gradient.addColorStop(1, "rgba(0,0,0,0)");
          ctx.strokeStyle = gradient;
          ctx.globalAlpha = lineAlpha;
          ctx.lineWidth = 1.5 * dpr;
          ctx.beginPath();
          ctx.moveTo(Math.max(w * 0.04, x - w * 0.12), mid);
          ctx.lineTo(Math.min(w * 0.96, x + w * 0.12), mid);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      } else {
        drawLine(ctx, w, h, dpr, lineAlpha, c.accent, amp);
      }
    }
    if (clip.peaks && profileMix > 0.01) drawProfile(ctx, w, h, dpr, profileMix, c, now, shake);
  }

  // ---- Small interactions ------------------------------------------------------
  document.addEventListener("click", (event) => {
    const swap = event.target.closest?.("#st-swap");
    if (swap && swap.dataset.auto !== "true" && !swap.disabled) swap.classList.toggle("st-turn");
  });
  // Ctrl/⌘ + Enter submits when the button is enabled.
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      const button = $("#st-submit");
      if (button && !button.disabled) { event.preventDefault(); button.click(); }
    }
  });

  setInterval(tick, 100);
  tick();
  setInterval(() => renderLabel(performance.now()), 100);
  requestAnimationFrame(draw);
}
