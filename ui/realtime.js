() => {
  const waitForPage = setInterval(() => {
    const root = document.querySelector('#rt-test');
    if (!root || root.dataset.bound) return;
    clearInterval(waitForPage);
    root.dataset.bound = 'true';
    const $ = id => root.querySelector('#' + id);
    const page = new URL(location.href);
    page.search = ''; page.hash = '';
    if (!page.pathname.endsWith('/')) page.pathname += '/';
    const base = new URL(root.dataset.apiBase || 'realtime-api/', page);
    let current = null;
    let initialized = false;
    let samples = [];
    let lastSnapshot = null, lastHistory = [];
    /* AUDIO_TRANSPORT */
    /* ENDPOINT_DETECTOR */
    const state = message => { $('rt-status').textContent = message; };
    // Append real new characters without replacing the whole text node each poll.
    const text = (id, value) => {
      const element = $(id), next = value || '', previous = element.textContent;
      if (next === previous) return;
      const panel = ['rt-transcript', 'rt-translation'].includes(id) ? element.closest('.rt-results > section') : null;
      const follow = panel && panel.scrollHeight - panel.scrollTop - panel.clientHeight < 40;
      if (follow) requestAnimationFrame(() => { panel.scrollTop = panel.scrollHeight; });
      const node = element.firstChild;
      if (!node || node.nodeType !== 3 || element.childNodes.length !== 1) { element.textContent = next; return; }
      if (next.startsWith(previous)) { node.appendData(next.slice(previous.length)); return; }
      let common = 0;
      while (common < Math.min(previous.length, next.length) && previous[common] === next[common]) common++;
      node.replaceData(common, previous.length - common, next.slice(common));
    };
    const controls = active => {
      $('rt-start').disabled = active || !initialized;
      $('rt-replay').disabled = active || !initialized;
      $('rt-stop').disabled = !active || !current?.node || current.stopping;
      $('rt-cancel').disabled = !active;
      for (const id of ['rt-source', 'rt-target', 'rt-sample', 'rt-transport', 'rt-playback-mode', 'rt-auto-stop', 'rt-endpoint-delay']) $(id).disabled = active;
    };
    async function api(path, payload, run) {
      const sent = performance.now(), wall = Date.now();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 30000);
      try {
        const response = await fetch(new URL(path, base), {
          method: payload === undefined ? 'GET' : 'POST', credentials: 'same-origin',
          headers: payload === undefined ? {} : {'Content-Type': 'application/json'},
          body: payload === undefined ? undefined : JSON.stringify(payload), signal: controller.signal,
        });
        const data = await response.json();
        if (run) {
          run.metrics.result_rtt_ms = performance.now() - sent;
          calibrate(run, data.server_time_ms, wall, Date.now(), run.metrics.result_rtt_ms);
        }
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        return data;
      } finally { clearTimeout(timer); }
    }
    function options() {
      return {source_lang: $('rt-source').value, target_lang: $('rt-target').value,
        };
    }
    function reset() {
      lastSnapshot = null; lastHistory = [];
      for (const id of ['rt-transcript','rt-translation','rt-completed-translation','rt-translation-draft','rt-draft-source','rt-draft-version','rt-history','rt-mt-source','rt-sync','rt-asr-version','rt-mt-version']) $(id).replaceChildren();
      $('rt-live-translation').hidden = true;
      root.querySelector('.rt-raw-generation').open = false;
      $('rt-diagnostics').textContent = '正在建立连接，计时从声音开始输入计算。';
    }
    function render(event) {
      const row = document.createElement('li');
      const kind = event.type === 'asr' ? '原文' : '译文';
      const version = event.type === 'asr' ? event.asr_version : event.translation_version;
      const status = event.type === 'asr' ? (event.is_final ? '最终' : '更新') : (event.is_final ? '检查点' : '草稿');
      row.textContent = `${event.observed_seconds}s · ${kind} v${version} · ${status}：${event.transcript ?? event.translation ?? ''}`;
      $('rt-history').append(row);
      while ($('rt-history').children.length > 200) $('rt-history').firstChild.remove();
    }
    function renderASR(run, snapshot) {
      const version = snapshot.asr_version;
      if (version < (run.asrVersion || 0) || (version === run.asrVersion && run.asrFinal && !snapshot.asr_final)) return false;
      run.asrVersion = version; run.asrFinal = Boolean(snapshot.asr_final);
      if (snapshot.first_asr_seconds != null) run.firstASRSeconds = snapshot.first_asr_seconds;
      if (snapshot.first_audio_seconds != null) run.firstAudioSeconds = snapshot.first_audio_seconds;
      text('rt-transcript', snapshot.transcript);
      $('rt-asr-version').textContent = `v${snapshot.asr_version}${snapshot.asr_final ? ' · 最终' : ' · 更新中'}`;
      return true;
    }
    function showASR(run, snapshot) {
      if (current !== run || !renderASR(run, snapshot) || run.inputStart === undefined) return;
      requestAnimationFrame(() => {
        if (snapshot.transcript && run.metrics.first_asr_display_seconds === undefined)
          run.metrics.first_asr_display_seconds = (performance.now() - run.inputStart) / 1000;
        if (snapshot.asr_server_time_ms != null && run.clockOffset !== undefined)
          run.metrics.asr_delivery_estimate_ms = Math.max(0, Date.now() + run.clockOffset - snapshot.asr_server_time_ms);
        diagnostics(run);
      });
    }
    function renderSnapshot(snapshot, pipeline, run) {
      lastSnapshot = snapshot;
      renderASR(run, snapshot);
      const displayed = snapshot.translation_display ?? snapshot.translation;
      const version = snapshot.translation_display_version ?? snapshot.translation_version;
      const asrVersion = run.asrVersion ?? snapshot.asr_version;
      const provisional = Boolean(snapshot.translation_display_provisional);
      text('rt-translation', displayed);
      text('rt-mt-source', snapshot.translation_display_source ?? snapshot.translation_source);
      if (root.querySelector('.rt-raw-generation').open) renderDebug(snapshot);
      $('rt-mt-version').textContent = pipeline === 'asr' ? 'ASR 单独测试' :
        `对应原文 v${version}${provisional ? ' · 尾部可修订' : ''}`;
      $('rt-sync').textContent = pipeline === 'asr' ? '本次仅测试识别，没有加载翻译模型。' :
        snapshot.translation_final && !provisional && version === asrVersion ? '最终原文和译文已同步。' :
        snapshot.translation_pending ? `字幕保留在屏幕上，正在跟进原文 v${asrVersion}；更新不会从头重新播放。` :
        '字幕已更新，继续讲话时尾部仍可修订。';
    }
    function renderDebug(snapshot) {
      const draft = Boolean(snapshot.translation_draft);
      $('rt-live-translation').hidden = !draft;
      text('rt-translation-draft', draft ? snapshot.translation_draft : '');
      text('rt-draft-source', draft ? snapshot.translation_draft_source : '');
      text('rt-completed-translation', snapshot.translation);
      $('rt-draft-version').textContent = draft ? `正在生成 · 原文 v${snapshot.translation_draft_version} · 尚未完成` : '';
    }
    async function release(run) {
      run.download?.abort();
      run.source?.disconnect(); run.stream?.getTracks().forEach(track => track.stop());
      run.node?.disconnect(); run.output?.disconnect();
      if (run.context && run.context.state !== 'closed') await run.context.close();
    }
    async function fail(run, message) {
      if (current !== run) return;
      run.failure = new Error(message); current = null;
      run.metrics.error = message; disposeTransport(run); report(run);
      await release(run);
      if (run.id) api('cancel', {session_id: run.id}).catch(() => {});
      state('测试停止：' + message); controls(false);
    }
    function showResults(run, result) {
        if (current !== run) return;
        run.lastResult = result;
        // Paint the latest text before rendering the diagnostic history.
        renderSnapshot(result.snapshot, result.pipeline, run);
        for (const event of result.events) {
          if (event.event_seq <= run.cursor) continue;
          if (event.event_seq !== run.cursor + 1) { fail(run, '结果事件序号不连续'); return; }
          run.cursor = event.event_seq;
        }
        run.history ||= [];
        run.history.push(...result.events.slice(-200));
        run.history = run.history.slice(-200);
        lastHistory = run.history;
        // Closed diagnostic history must not create hundreds of DOM rows during
        // live ASR. Render it only when the user opens it.
        if (root.querySelector('.rt-history').open) {
          $('rt-history').replaceChildren();
          for (const event of run.history) render(event);
        }
        if (run.inputStart !== undefined) requestAnimationFrame(() => {
          const elapsed = (performance.now() - run.inputStart) / 1000;
          if (result.snapshot.transcript && run.metrics.first_asr_display_seconds === undefined) run.metrics.first_asr_display_seconds = elapsed;
          if ((result.snapshot.translation_display ?? result.snapshot.translation) && run.metrics.first_translation_display_seconds === undefined) run.metrics.first_translation_display_seconds = elapsed;
          if (result.snapshot.translation && run.metrics.first_complete_translation_display_seconds === undefined) run.metrics.first_complete_translation_display_seconds = elapsed;
          const event = result.events[result.events.length - 1];
          if (event && run.clockOffset !== undefined) run.metrics.delivery_estimate_ms = Math.max(0, Date.now() + run.clockOffset - event.server_time_ms);
          if (result.snapshot.translation_display_seq !== run.lastDisplaySeq && result.snapshot.translation_display_server_time_ms != null && run.clockOffset !== undefined) {
            run.metrics.delivery_estimate_ms = Math.max(0, Date.now() + run.clockOffset - result.snapshot.translation_display_server_time_ms);
            run.lastDisplaySeq = result.snapshot.translation_display_seq;
          }
          diagnostics(run);
        });
        const playback = run.kind === 'replay' ? `已播放 ${(run.played || 0).toFixed(1)} 秒 · ` : '';
        $('rt-stats').textContent = `${run.transport} · ${playback}待上传确认 ${(run.upload?.pending || 0).toFixed(1)} 秒 · 已接收 ${result.received_seconds.toFixed(1)} 秒 · 已处理 ${result.processed_seconds.toFixed(1)} 秒 · 待处理 ${result.queued_seconds.toFixed(1)} 秒 · 检查点待处理 ${result.pending_checkpoints} 条`;
        if (result.status === 'error' || result.status === 'cancelled') {
          fail(run, result.error || result.status); return;
        }
        if (result.status === 'finished') {
          current = null; disposeTransport(run); release(run); report(run); controls(false);
          state(result.snapshot.transcript ? `${run.stopReason ? run.stopReason + '；' : ''}处理完成，最终结果已同步。可以开始下一次测试。` : '处理完成，未识别到文字。请检查麦克风或换一个示例。');
          return;
        }
        state(run.stopping ? `${run.stopReason || '已停止输入'}，正在补齐最终原文和译文…` : run.pauseWanted ? '网络适配模式暂缓：等待上传确认后自动继续…' : run.kind === 'replay' ? '音频按真实时钟播放，原文先显示，译文随后跟进…' : '正在持续录音，原文与译文分别更新…');
    }
    async function poll(run) {
      if (current !== run || run.transport !== 'http') return;
      const started = performance.now();
      try {
        const result = await api(`results?session_id=${encodeURIComponent(run.id)}&cursor=${run.cursor}`, undefined, run);
        showResults(run, result);
      } catch (error) { await fail(run, '获取结果失败：' + error.message); return; }
      run.pollTimer = setTimeout(() => poll(run), Math.max(10, 100 - (performance.now() - started)));
    }
    function send(run, audio) {
      if (current !== run || run.failure) return;
      run.upload.add(audio);
      // Enqueue this block first. finishInput then flushes the worklet tail and
      // waits for every ACK before asking the backend to finalize ASR and MT.
      if (!run.stopping && run.kind === 'microphone' && run.endpoint?.feed(audio)) {
        run.stopReason = `已检测到讲话后静音 ${run.endpoint.limit} 秒，自动停止录音`;
        run.metrics.endpoint_silence_seconds = run.endpoint.limit;
        finishInput(run);
      }
    }
    function calibrate(run, serverTime, sentWall, receivedWall, rtt) {
      if (!Number.isFinite(serverTime) || (run.metrics.clock_rtt_ms !== undefined && rtt >= run.metrics.clock_rtt_ms)) return;
      run.metrics.clock_rtt_ms = rtt; run.clockOffset = serverTime - (sentWall + receivedWall) / 2;
    }
    function diagnostics(run) {
      const number = value => Number.isFinite(value) ? value.toFixed(2) : '等待';
      const snap = run.lastResult?.snapshot || {}, first = run.firstAudioSeconds ?? snap.first_audio_seconds;
      const firstASR = run.firstASRSeconds ?? snap.first_asr_seconds;
      $('rt-diagnostics').textContent = [
        `传输：${run.transport || '连接中'}${run.metrics.fallback_reason ? '；' + run.metrics.fallback_reason : ''}`,
        `开播/采集 → 首条原文可见：${number(run.metrics.first_asr_display_seconds)} 秒；首个译文片段可见：${number(run.metrics.first_translation_display_seconds)} 秒；首份完整译文可见：${number(run.metrics.first_complete_translation_display_seconds)} 秒`,
        `后端收到首块 → 首条原文产生：${number(firstASR != null && first != null ? firstASR - first : undefined)} 秒；首个译文片段产生：${number(snap.first_translation_draft_seconds != null && first != null ? snap.first_translation_draft_seconds - first : undefined)} 秒；首份完整译文产生：${number(snap.first_translation_seconds != null && first != null ? snap.first_translation_seconds - first : undefined)} 秒`,
        `上传确认 RTT：${number(run.metrics.upload_rtt_ms)} ms；HTTP 查询 RTT：${number(run.metrics.result_rtt_ms)} ms`,
        `最近结果产生 → 页面可见（估计）：${number(run.metrics.delivery_estimate_ms)} ms；校时 RTT：${number(run.metrics.clock_rtt_ms)} ms`,
        `ASR 独立推送 → 原文可见（估计）：${number(run.metrics.asr_delivery_estimate_ms)} ms；播放节奏：${run.playbackMode === 'adaptive' ? '网络适配' : '实时，不等字幕'}`,
        `最大未确认音频：${number(run.upload?.maxPending)} 秒；回放自动暂缓：${run.metrics.playback_pauses || 0} 次`,
        ...(run.stopReason ? ['输入结束原因：' + run.stopReason] : []),
        ...(run.metrics.error ? ['停止原因：' + run.metrics.error] : []),
        '跨设备校时估计受网络不对称影响，不是逐词的准确延迟。',
      ].join('\n');
    }
    function report(run) {
      run.metrics.max_pending_seconds = run.upload?.maxPending || 0;
      diagnostics(run);
      if (run.id) api('diagnostics', {session_id: run.id, transport: run.transport, ...run.metrics}).catch(() => {});
    }
    function disposeTransport(run) {
      clearTimeout(run.pollTimer); clearInterval(run.pingTimer); clearInterval(run.watchTimer);
      run.upload?.close();
      if (run.socket) { run.socket.ignored = true; run.socket.close(); }
    }
    function pressure(run) {
      if (run.kind !== 'replay' || run.playbackMode !== 'adaptive' || current !== run || run.stopping) return;
      const wanted = Boolean(run.uploadPressure);
      if (wanted === Boolean(run.pauseWanted)) return;
      run.pauseWanted = wanted;
      if (wanted) run.metrics.playback_pauses = (run.metrics.playback_pauses || 0) + 1;
      run.pauseTask = (run.pauseTask || Promise.resolve()).then(async () => {
        if (current !== run || run.stopping || run.context.state === 'closed') return;
        if (run.pauseWanted) await run.context.suspend(); else await run.context.resume();
      }).catch(error => fail(run, '回放暂停/恢复失败：' + error.message));
    }
    async function connect(run, session) {
      run.metrics ||= {}; run.transport = 'http';
      if ($('rt-transport').value !== 'http' && window.WebSocket && session.stream_ticket) {
        const url = new URL('stream', base); url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(url); run.socket = socket;
        const ready = await new Promise(resolve => {
          let done = false;
          const settle = ok => { if (done) return; done = true; clearTimeout(timer); resolve(ok); };
          const timer = setTimeout(() => settle(false), 3000);
          socket.onopen = () => socket.send(JSON.stringify({session_id: run.id, ticket: session.stream_ticket, asr_updates: true}));
          socket.onmessage = event => {
            if (socket.ignored || current !== run) return;
            try {
              const message = JSON.parse(event.data);
              if (message.type === 'ready') { run.transport = 'websocket'; settle(true); }
              else if (message.type === 'asr') showASR(run, message.snapshot);
              else if (message.type === 'results') showResults(run, message.result);
              else if (message.type === 'ack') run.upload?.ack(message.acked_seq);
              else if (message.type === 'pong') calibrate(run, message.sent_server_time_ms, message.client_time_ms, Date.now(), Date.now() - message.client_time_ms);
              else if (message.type === 'error') { if (!done) settle(false); else fail(run, message.message); }
            } catch (error) { fail(run, '实时消息处理失败：' + error.message); }
          };
          socket.onerror = () => { if (!done) settle(false); };
          socket.onclose = event => {
            if (!done) settle(false);
            else if (!socket.ignored && current === run && run.transport === 'websocket') fail(run, `实时连接中断（${event.code}），请重试；诊断记录已保留`);
          };
        });
        if (!ready) {
          run.transport = 'http';
          socket.ignored = true; socket.close();
          run.metrics.fallback_reason = '长连接未能建立，自动回退 HTTP 合并上传';
        }
      }
      if (current !== run) return;
      run.upload = new AudioUploadQueue({rate: run.rate, mode: run.transport, now: () => performance.now(),
        maxPendingSeconds: run.kind === 'replay' && run.playbackMode !== 'adaptive' ? Math.max(8, Math.min(120, run.duration + .5)) : 8,
        socketSend: packet => run.socket.send(packet),
        httpSend: packet => {
          const bytes = packet.bytes.subarray(8); let binary = '';
          for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
          return api('audio', {session_id: run.id, chunk_seq: packet.sequence, sample_rate: run.rate, pcm16: btoa(binary)});
        },
        onPressure: value => { run.uploadPressure = value; pressure(run); },
        onError: error => fail(run, '上传失败：' + error.message), onAck: () => { run.lastAck = performance.now(); },
        onRtt: value => { run.metrics.upload_rtt_ms = value; },
      });
      if (run.transport === 'websocket') {
        run.lastAck = performance.now();
        const ping = () => { if (current === run && run.socket.readyState === 1) run.socket.send(JSON.stringify({type: 'ping', client_time_ms: Date.now()})); };
        ping(); run.pingTimer = setInterval(ping, 2000);
        run.watchTimer = setInterval(() => {
          if (current === run && run.upload.inflight.size && performance.now() - run.lastAck > 30000)
            fail(run, '实时上传超过 30 秒未获确认；连接可能中断，诊断记录已保留');
        }, 1000);
      } else poll(run);
    }
    const processorCode = `
      class Capture extends AudioWorkletProcessor {
        constructor() {
          super(); this.samples = new Float32Array(Math.round(sampleRate * 0.16)); this.offset = 0; this.stopped = false;
          this.port.onmessage = () => {
            this.stopped = true;
            if (this.offset) this.port.postMessage({audio: this.samples.slice(0, this.offset)});
            this.port.postMessage({flushed: true});
          };
        }
        process(inputs) {
          if (this.stopped) return false;
          const channels = inputs[0];
          if (!channels?.length) return true;
          for (let i = 0; i < channels[0].length; i++) {
            let value = 0; for (const channel of channels) value += channel[i];
            this.samples[this.offset++] = value / channels.length;
            if (this.offset === this.samples.length) {
              const audio = this.samples; this.port.postMessage({audio}, [audio.buffer]);
              this.samples = new Float32Array(Math.round(sampleRate * 0.16)); this.offset = 0;
            }
          }
          return true;
        }
      }
      registerProcessor('rt-capture', Capture);
      class Replay extends AudioWorkletProcessor {
        constructor() {
          super(); this.audio = null; this.position = 0; this.offset = 0; this.stopped = false;
          this.samples = new Float32Array(Math.round(sampleRate * 0.16));
          this.port.onmessage = event => {
            if (event.data.audio && !this.stopped) this.audio = event.data.audio;
            else if (event.data.flush) this.finish();
          };
        }
        finish() {
          if (this.stopped) return;
          this.stopped = true;
          if (this.offset) this.port.postMessage({audio: this.samples.slice(0, this.offset), playedFrames: this.position});
          this.port.postMessage({flushed: true});
          this.port.postMessage({ended: true});
        }
        process(inputs, outputs) {
          if (this.stopped) return false;
          if (!this.audio) return true;
          const output = outputs[0][0];
          const count = Math.min(output.length, this.audio.length - this.position);
          // Playback and upload use exactly the same samples and rendering clock.
          for (let i = 0; i < count; i++) {
            const value = this.audio[this.position++]; output[i] = value;
            this.samples[this.offset++] = value;
            if (this.offset === this.samples.length) {
              const audio = this.samples;
              this.port.postMessage({audio, playedFrames: this.position}, [audio.buffer]);
              this.samples = new Float32Array(Math.round(sampleRate * 0.16)); this.offset = 0;
            }
          }
          if (this.position === this.audio.length) this.finish();
          return !this.stopped;
        }
      }
      registerProcessor('rt-replay', Replay);
    `;
    async function makeNode(run, name) {
      const blob = URL.createObjectURL(new Blob([processorCode], {type: 'text/javascript'}));
      try { await run.context.audioWorklet.addModule(blob); } finally { URL.revokeObjectURL(blob); }
      if (current !== run) return;
      run.node = new AudioWorkletNode(run.context, name,
        name === 'rt-replay' ? {numberOfInputs: 0, outputChannelCount: [1]} : {});
      run.node.port.onmessage = event => {
        if (current !== run) return;
        if (event.data.audio) send(run, event.data.audio);
        if (event.data.playedFrames !== undefined) {
          run.played = event.data.playedFrames / run.rate;
          $('rt-playback-progress').value = run.played;
          $('rt-playback-time').textContent = `${run.played.toFixed(1)} / ${run.duration.toFixed(1)} 秒`;
        }
        if (event.data.flushed) { run.flushed = true; run.flushResolve?.(); }
        if (event.data.ended) finishInput(run);
      };
      run.node.onprocessorerror = () => fail(run, '浏览器音频处理失败，请刷新后重试');
    }
    async function finishInput(run) {
      if (current !== run || !run.node || run.stopping) return;
      run.stopping = true; controls(true); state(`${run.stopReason || '已停止录音'}，正在确认最后的音频块…`);
      try {
        run.source?.disconnect();
        await run.pauseTask;
        if (!run.flushed && run.context.state === 'suspended') await run.context.resume();
        if (!run.flushed) await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('尾部音频确认超时')), 3000);
          run.flushResolve = () => { clearTimeout(timer); resolve(); };
          run.node.port.postMessage({flush: true});
        });
        // Stop microphone capture immediately. For replay, keep the output open
        // until the final upload is acknowledged so the audible tail can drain.
        if (run.kind === 'microphone') await release(run);
        await run.upload.wait();
        if (run.kind === 'replay') await release(run);
        if (current !== run || run.failure) return;
        if (run.transport === 'websocket') run.socket.send(JSON.stringify({type: 'finish', last_chunk_seq: run.upload.sequence}));
        else await api('finish', {session_id: run.id, last_chunk_seq: run.upload.sequence});
      } catch (error) { await fail(run, error.message); }
    }
    function setVolume() {
      if (current?.output) current.output.gain.value = $('rt-mute').checked ? 0 : Number($('rt-volume').value);
    }
    $('rt-volume').oninput = setVolume; $('rt-mute').onchange = setVolume;
    $('rt-start').onclick = async () => {
      if (current) return;
      const run = {kind: 'microphone', cursor: 0, metrics: {}};
      current = run; reset(); controls(true); state('准备麦克风…');
      $('rt-playback').hidden = true;
      try {
        if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('需要 HTTPS 和支持 AudioWorklet 的现代浏览器');
        run.stream = await navigator.mediaDevices.getUserMedia({audio: true});
        if (current !== run) { await release(run); return; }
        run.context = new AudioContext(); run.rate = run.context.sampleRate;
        if ($('rt-auto-stop').checked) run.endpoint = new AudioEndpointDetector({
          rate: run.rate, silenceSeconds: Number($('rt-endpoint-delay').value),
        });
        await makeNode(run, 'rt-capture');
        if (current !== run) return;
        const session = await api('start', options()); run.id = session.session_id;
        if (current !== run) { await release(run); api('cancel', {session_id: run.id}).catch(() => {}); return; }
        await connect(run, session);
        if (current !== run) { api('cancel', {session_id: run.id}).catch(() => {}); return; }
        run.inputStart = performance.now();
        run.source = run.context.createMediaStreamSource(run.stream);
        const mute = run.context.createGain(); mute.gain.value = 0;
        run.source.connect(run.node); run.node.connect(mute); mute.connect(run.context.destination);
        await run.context.resume(); controls(true);
      } catch (error) { await fail(run, error.message); }
    };
    $('rt-stop').onclick = async () => {
      if (current) await finishInput(current);
    };
    $('rt-cancel').onclick = () => { if (current) fail(current, '已主动取消本次测试。'); };
    $('rt-replay').onclick = async () => {
      if (current) return;
      const run = {kind: 'replay', cursor: 0, played: 0, playbackMode: $('rt-playback-mode').value, metrics: {}};
      current = run; reset(); controls(true); state('正在加载测试音频…');
      $('rt-playback').hidden = true;
      try {
        if (!window.AudioWorkletNode) throw new Error('声音同步测试需要 HTTPS 和支持 AudioWorklet 的现代浏览器');
        const index = Number($('rt-sample').value), sample = samples.find(item => item.id === index);
        if (!sample?.available) throw new Error('请选择存在且未超过输入时长上限的示例');
        // Unlock audio inside the click gesture, before network/decode awaits.
        run.context = new AudioContext(); run.rate = run.context.sampleRate;
        await run.context.resume();
        if (run.context.state !== 'running') throw new Error('浏览器没有允许声音播放，请允许此页面播放声音后重试');
        run.download = new AbortController();
        const timer = setTimeout(() => run.download.abort(), 30000);
        let buffer;
        try {
          const response = await fetch(new URL(`samples/${index}/audio?v=${encodeURIComponent(sample.version || '')}`, base),
            {credentials: 'same-origin', signal: run.download.signal});
          if (!response.ok) throw new Error(`音频下载失败：HTTP ${response.status}`);
          buffer = await run.context.decodeAudioData(await response.arrayBuffer());
        } finally { clearTimeout(timer); }
        if (current !== run) return;
        // Mono samples are shared by the audible output and PCM16 upload.
        const mono = new Float32Array(buffer.length);
        for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
          const audio = buffer.getChannelData(channel);
          for (let i = 0; i < mono.length; i++) mono[i] += audio[i] / buffer.numberOfChannels;
        }
        run.duration = mono.length / run.rate;
        await makeNode(run, 'rt-replay');
        if (current !== run) return;
        $('rt-source').value = sample.source_lang;
        $('rt-target').value = sample.target_lang;
        const session = await api('start', options()); run.id = session.session_id;
        if (current !== run) { api('cancel', {session_id: run.id}).catch(() => {}); return; }
        await connect(run, session);
        if (current !== run) { api('cancel', {session_id: run.id}).catch(() => {}); return; }
        $('rt-playback').hidden = false; $('rt-playback-progress').max = run.duration;
        $('rt-playback-progress').value = 0; $('rt-playback-time').textContent = `0.0 / ${run.duration.toFixed(1)} 秒`;
        run.output = run.context.createGain(); setVolume();
        run.node.connect(run.output); run.output.connect(run.context.destination);
        run.inputStart = performance.now();
        run.node.port.postMessage({audio: mono}, [mono.buffer]);
        controls(true);
      } catch (error) { await fail(run, error.message); }
    };
    window.addEventListener('pagehide', () => {
      const run = current;
      if (run?.id) navigator.sendBeacon(new URL('cancel', base), new Blob([JSON.stringify({session_id: run.id})], {type: 'application/json'}));
      if (run) { disposeTransport(run); release(run); }
    });
    root.querySelector('.rt-history').ontoggle = () => {
      if (!root.querySelector('.rt-history').open) return;
      $('rt-history').replaceChildren();
      for (const event of lastHistory) render(event);
    };
    root.querySelector('.rt-raw-generation').ontoggle = () => {
      if (root.querySelector('.rt-raw-generation').open && lastSnapshot) renderDebug(lastSnapshot);
    };
    controls(false);
    api('meta').then(data => {
      if (data.languages) for (const [id, selected] of [['rt-source', 'cmn'], ['rt-target', 'eng']]) {
        $(id).replaceChildren();
        const languages = id === 'rt-source' ? data.source_languages : data.languages;
        for (const [label, code] of languages) { const option = new Option(label, code); $(id).append(option); }
        $(id).value = selected;
      }
      samples = data.samples || [];
      $('rt-sample').replaceChildren();
      for (const sample of samples) {
        const option = new Option(sample.label + (sample.available ? '' : '（不可用、语言不支持或超过时长上限）'), sample.id);
        option.disabled = !sample.available; $('rt-sample').append(option);
      }
      const preferred = samples.find(item => item.available);
      if (preferred) $('rt-sample').value = String(preferred.id);
      $('rt-stats').textContent = `麦克风需要 HTTPS 与浏览器授权。最多 ${data.max_seconds} 秒，同时一个会话。`;
      $('rt-pipeline').textContent = data.pipeline === 'asr' ? 'Nemotron 流式 ASR 单独测试' : `Nemotron ASR + ${data.translation_backend}`; initialized = true; controls(false);
      state(data.backend.includes('MOCK') ? 'MOCK 预览：仅验证录音和版本同步流程，不进行识别或翻译。' : '模型已就绪，可以开始录音或回放示例。');
    }).catch(error => state('初始化失败：' + error.message));
  }, 100);
}
