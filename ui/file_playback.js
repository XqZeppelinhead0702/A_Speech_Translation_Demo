// A separate player survives Gradio locking/rebuilding the upload widget.
// start() is called in the submit gesture, before any model/network waits.
const filePlayback = (() => {
  let source = '', started = -Infinity, generation = 0;
  const audio = () => document.querySelector('#st-playback-audio');
  const panel = () => document.querySelector('#st-file-playback');
  const note = message => { const el = document.querySelector('#st-playback-note'); if (el) el.textContent = message; };
  const url = clip => clip?.url || document.querySelector('#st-audio audio')?.currentSrc || document.querySelector('#st-audio audio')?.src;
  const stop = () => {
    generation++;
    const player = audio();
    if (player) { player.pause(); player.removeAttribute('src'); player.load(); }
    if (panel()) panel().hidden = true;
    source = ''; started = -Infinity;
  };
  const start = clip => {
    const src = url(clip), player = audio(), container = panel();
    if (!src || !player || !container) return;
    // The capture listener and Gradio's submit JS may both run for one click.
    if (source === src && performance.now() - started < 1000) return;
    const token = ++generation;
    source = src; started = performance.now();
    document.querySelector('#st-audio audio')?.pause();
    container.hidden = false; note('输入音频试听 · 识别与翻译同时进行');
    player.onended = () => note('输入音频已播放完毕');
    player.onerror = () => { if (token === generation) note('试听加载失败，可使用原音频播放器；翻译继续进行'); };
    try {
      if (player.getAttribute('src') !== src) player.src = src;
      player.currentTime = 0;
      Promise.resolve(player.play()).catch(() => {
        if (token === generation) note('浏览器未允许自动播放，请点击下方播放键；翻译继续进行');
      });
    } catch (error) { note('请点击下方播放键试听；翻译继续进行'); }
  };
  return {start, stop, changed: clip => {
    if (!clip || (clip.url && source && clip.url !== source)) stop();
  }};
})();
window.stFilePlayback = filePlayback;
document.addEventListener('click', event => {
  const target = event.target;
  if (target.closest?.('#st-submit:not(:disabled)')) filePlayback.start(null);
  if (target.closest?.('#st-playback-stop, #st-clear, .st-example, #rt-start, #rt-replay, #st-tabs button[role="tab"]')) filePlayback.stop();
}, true);
// Never mix the automatic preview with manual playback of the upload widget.
document.addEventListener('play', event => {
  if (event.target.closest?.('#st-audio')) filePlayback.stop();
}, true);
window.addEventListener('pagehide', filePlayback.stop);
