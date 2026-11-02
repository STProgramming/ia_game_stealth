// Short procedural effects: every sound comes from a server event, with no
// asset download and no hidden position information used by the audio layer.
export class Soundscape {
  constructor() {
    this.enabled = true;
    this.seen = new Set();
    this.session = "";
  }

  async unlock() {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) return;
    if (!this.context) {
      this.context = new Audio();
      this.master = this.context.createGain(); this.master.gain.value = .4;
      this.master.connect(this.context.destination);
    }
    if (this.context.state === "suspended") await this.context.resume();
  }

  update(state) {
    const session = `${state.revision}:${state.mode}:${state.level}:${state.scenario}`;
    if (session !== this.session) { this.seen.clear(); this.session = session; }
    for (const event of state.noise || []) {
      if (this.seen.has(event.id)) continue;
      this.seen.add(event.id);
      if (!this.enabled || !this.context || this.context.state !== "running") continue;
      const dx = event.x - (state.player?.x || 0), dy = event.y - (state.player?.y || 0);
      const distance = Math.hypot(dx, dy), radius = event.radius || 5;
      if (distance > radius) continue;
      const strength = Math.max(.04, 1 - distance / radius) / (1 + distance * .13);
      this.play(event.source, strength, Math.max(-1, Math.min(1, dx / 10)));
    }
    if (this.seen.size > 4096) this.seen = new Set([...this.seen].slice(-2048));
  }

  play(source, strength, pan) {
    const profiles = {
      GUNSHOT: [.17, .22, 1800], "SUPPRESSED SHOT": [.07, .07, 520], "SUBSONIC SHOT": [.045, .028, 260],
      "BULLET IMPACT": [.065, .09, 1300], "MAGAZINE DROP": [.11, .07, 700],
      "CASING DROP": [.1, .025, 2200], RELOAD: [.12, .025, 900],
    };
    const profile = profiles[source];
    if (!profile) return;
    const [duration, volume, frequency] = profile, ctx = this.context, now = ctx.currentTime;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(Math.max(.0001, volume * strength), now);
    gain.gain.exponentialRampToValueAtTime(.0001, now + duration);
    const panner = ctx.createStereoPanner(); panner.pan.value = pan;
    gain.connect(panner); panner.connect(this.master);
    if (source === "CASING DROP") {
      const oscillator = ctx.createOscillator(); oscillator.type = "triangle";
      oscillator.frequency.setValueAtTime(frequency, now);
      oscillator.frequency.exponentialRampToValueAtTime(900, now + duration);
      oscillator.connect(gain); oscillator.start(now); oscillator.stop(now + duration);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); panner.disconnect(); };
    } else {
      const buffer = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * duration), ctx.sampleRate);
      const samples = buffer.getChannelData(0);
      for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;
      const noise = ctx.createBufferSource(); noise.buffer = buffer;
      const filter = ctx.createBiquadFilter(); filter.type = "lowpass"; filter.frequency.value = frequency;
      noise.connect(filter); filter.connect(gain); noise.start(now);
      noise.onended = () => { noise.disconnect(); filter.disconnect(); gain.disconnect(); panner.disconnect(); };
    }
  }
}

if (typeof window !== "undefined") {
  const audio = new Soundscape(); window.shadowAudio = audio;
  const unlock = () => { audio.unlock().catch(() => {}); };
  document.addEventListener("pointerdown", unlock, {capture: true});
  document.addEventListener("keydown", unlock, {capture: true});
  const button = document.getElementById("soundBtn");
  button?.addEventListener("click", () => {
    audio.enabled = !audio.enabled;
    button.textContent = `AUDIO ${audio.enabled ? "ON" : "OFF"}`;
    button.setAttribute("aria-pressed", String(audio.enabled));
    document.getElementById("game").focus({preventScroll: true});
  });
}
