"use strict";
// Home games — synthesized table sounds (WebAudio, no audio files).
// Every sound is a few oscillators / a filtered noise burst, so there is
// nothing to download and nothing to license. Audio starts only after the
// first user gesture (browser autoplay policy) — `unlock()` is wired to the
// first pointerdown / keydown by games.js.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  let ctx = null;
  let master = null;
  let enabled = true;
  let volume = 0.6;
  let noiseBuf = null;

  function ensure() {
    if (ctx) return ctx;
    const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AC) return null;
    try {
      ctx = new AC();
      master = ctx.createGain();
      master.gain.value = volume;
      master.connect(ctx.destination);
      const len = Math.floor(ctx.sampleRate * 0.6);
      noiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    } catch (_) {
      ctx = null;
    }
    return ctx;
  }

  function unlock() {
    const c = ensure();
    if (c && c.state === "suspended") c.resume().catch(() => {});
  }

  // One enveloped oscillator. `to` glides the pitch.
  function tone(t0, freq, dur, gain, type, to) {
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = type || "sine";
    o.frequency.setValueAtTime(freq, t0);
    if (to) o.frequency.exponentialRampToValueAtTime(to, t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + 0.006);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g).connect(master);
    o.start(t0);
    o.stop(t0 + dur + 0.02);
  }

  // A band-passed noise burst (card swish, chip clack body).
  function noise(t0, dur, gain, freq, q) {
    const s = ctx.createBufferSource();
    s.buffer = noiseBuf;
    const f = ctx.createBiquadFilter();
    f.type = "bandpass";
    f.frequency.value = freq;
    f.Q.value = q || 1;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    s.connect(f).connect(g).connect(master);
    s.start(t0, Math.random() * 0.3);
    s.stop(t0 + dur + 0.02);
  }

  function clack(t, g) {
    // ceramic chip: a very short bright ping over a click of noise
    tone(t, 3100 + Math.random() * 500, 0.045, 0.16 * g, "triangle");
    tone(t, 1750 + Math.random() * 250, 0.06, 0.1 * g, "sine");
    noise(t, 0.03, 0.22 * g, 4200, 1.2);
  }

  const SOUNDS = {
    deal(t) { noise(t, 0.09, 0.2, 2600, 0.8); },
    flip(t) { noise(t, 0.05, 0.22, 3400, 1.4); tone(t, 520, 0.05, 0.05, "triangle", 260); },
    chip(t) { clack(t, 1); clack(t + 0.055, 0.8); },
    chips(t) { for (let i = 0; i < 5; i++) clack(t + i * 0.045 + Math.random() * 0.01, 1 - i * 0.12); },
    check(t) { tone(t, 150, 0.08, 0.5, "sine", 90); tone(t + 0.11, 140, 0.08, 0.42, "sine", 85); noise(t, 0.03, 0.12, 700, 1); noise(t + 0.11, 0.03, 0.1, 700, 1); },
    fold(t) { noise(t, 0.16, 0.13, 1500, 0.6); },
    turn(t) { tone(t, 660, 0.22, 0.2, "sine"); tone(t + 0.13, 990, 0.34, 0.2, "sine"); },
    tick(t) { tone(t, 1200, 0.035, 0.12, "square"); },
    urgent(t) { tone(t, 880, 0.09, 0.16, "square"); tone(t + 0.14, 880, 0.09, 0.16, "square"); },
    allin(t) { tone(t, 220, 0.5, 0.2, "sawtooth", 660); for (let i = 0; i < 7; i++) clack(t + 0.1 + i * 0.04, 0.9); },
    win(t) { [523, 659, 784, 1047].forEach((f, i) => tone(t + i * 0.09, f, 0.4, 0.17, "triangle")); for (let i = 0; i < 8; i++) clack(t + 0.2 + i * 0.05, 0.7); },
    pot(t) { for (let i = 0; i < 6; i++) clack(t + i * 0.04, 0.75); },
    msg(t) { tone(t, 880, 0.09, 0.1, "sine", 1320); },
    sit(t) { tone(t, 440, 0.12, 0.13, "sine"); tone(t + 0.1, 554, 0.18, 0.13, "sine"); },
    error(t) { tone(t, 220, 0.18, 0.16, "square", 150); },
  };

  HG.sound = {
    unlock,
    play(name) {
      if (!enabled || !SOUNDS[name]) return;
      const c = ensure();
      if (!c || c.state !== "running") return;
      try { SOUNDS[name](c.currentTime + 0.005); } catch (_) { /* never break the table */ }
    },
    setEnabled(on) { enabled = !!on; },
    setVolume(v) {
      volume = Math.max(0, Math.min(1, Number(v) || 0));
      if (master) master.gain.value = volume;
    },
    isEnabled() { return enabled; },
  };
})();
