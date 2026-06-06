// 効果音（Web Audio、アセット不要）。キャリブのフィードバックに使う。
// AudioContext はユーザー操作後に resume する必要があるため、resume() を gesture で呼ぶ。

export class Beeper {
  private ctx: AudioContext | null = null;

  private get audio(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    return this.ctx;
  }

  resume(): void {
    void this.ctx?.resume();
  }

  private tone(freq: number, durMs: number, gain = 0.06, type: OscillatorType = "sine"): void {
    const ctx = this.audio;
    if (ctx.state === "suspended") {
      void ctx.resume();
    }
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    osc.connect(amp);
    amp.connect(ctx.destination);
    const t = ctx.currentTime;
    amp.gain.setValueAtTime(gain, t);
    amp.gain.exponentialRampToValueAtTime(0.0001, t + durMs / 1000);
    osc.start(t);
    osc.stop(t + durMs / 1000);
  }

  ready(): void {
    this.tone(520, 80, 0.035);
  }

  captured(): void {
    this.tone(880, 130, 0.07);
  }

  complete(): void {
    this.tone(660, 130, 0.07);
    setTimeout(() => this.tone(990, 220, 0.07), 130);
  }
}
