// 固視（fixation）検出（純粋）。I-DT（dispersion threshold）方式。
// 直近ウィンドウ内の視線点の拡散（dispersion）がしきい値未満なら「固視中＝安定」。
// 入力は頭部不変の eye-in-head 視線なので、固視中に頭を少し動かしても安定判定は崩れない。

export class FixationDetector {
  private buf: { x: number; y: number; t: number }[] = [];

  constructor(
    private readonly windowSec = 0.2,
    private readonly threshold = 0.06,
  ) {}

  // 視線点を時刻 t(秒)で投入し、現在固視中かを返す。
  push(x: number, y: number, t: number): boolean {
    this.buf.push({ x, y, t });
    const cutoff = t - this.windowSec;
    while (this.buf.length > 0 && this.buf[0].t < cutoff) {
      this.buf.shift();
    }
    if (this.buf.length < 3) {
      return false;
    }
    return this.dispersion() < this.threshold;
  }

  // I-DT 拡散: (xmax-xmin)+(ymax-ymin)。
  dispersion(): number {
    if (this.buf.length < 2) {
      return Number.POSITIVE_INFINITY;
    }
    const xs = this.buf.map((p) => p.x);
    const ys = this.buf.map((p) => p.y);
    return Math.max(...xs) - Math.min(...xs) + (Math.max(...ys) - Math.min(...ys));
  }

  // 安定度に応じた重み（0.1–1.0）。固視がタイトなほど高い。
  quality(): number {
    const d = this.dispersion();
    return Math.min(1, Math.max(0.1, 1 - d / this.threshold));
  }

  reset(): void {
    this.buf = [];
  }
}

// 適応しきい値（純粋）。直近ウィンドウの dispersion 分布のパーセンタイルを「ユーザー個人が
// 達成できる安定度」として返す。固定マジックナンバーを避け、カメラ/照明に依らず初回から機能する。
export class AdaptiveThreshold {
  private buf: { v: number; t: number }[] = [];

  constructor(
    private readonly windowSec = 6,
    private readonly percentile = 0.55,
    private readonly minSamples = 20,
  ) {}

  push(v: number, t: number): void {
    this.buf.push({ v, t });
    const cutoff = t - this.windowSec;
    while (this.buf.length > 0 && this.buf[0].t < cutoff) {
      this.buf.shift();
    }
  }

  // 十分なサンプルが貯まるまでは Infinity（寛容）。
  value(): number {
    if (this.buf.length < this.minSamples) {
      return Number.POSITIVE_INFINITY;
    }
    const sorted = this.buf.map((b) => b.v).sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.floor(this.percentile * sorted.length));
    return sorted[idx];
  }

  reset(): void {
    this.buf = [];
  }
}
