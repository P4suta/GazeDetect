// One-Euro フィルタ（純粋）。低遅延かつ低ジッタ。視線点 PoR の平滑化に使う。
// 参考: Casiez et al. "1€ Filter"。

export class OneEuroFilter {
  private xPrev: number | null = null;
  private dxPrev = 0;
  private tPrev = 0;

  constructor(
    private readonly minCutoff = 1.0,
    private readonly beta = 0.0,
    private readonly dCutoff = 1.0,
  ) {}

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  lastValue(): number | null {
    return this.xPrev;
  }

  // x を時刻 t(秒) で平滑化して返す。
  filter(x: number, t: number): number {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = t;
      return x;
    }
    const dt = Math.max(t - this.tPrev, 1e-3);
    this.tPrev = t;

    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxHat;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat;
    return xHat;
  }

  reset(): void {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = 0;
  }
}

// 2D 点用のラッパ。
export class PointFilter {
  private fx: OneEuroFilter;
  private fy: OneEuroFilter;

  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.fx = new OneEuroFilter(minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter(minCutoff, beta, dCutoff);
  }

  filter(x: number, y: number, t: number): { x: number; y: number } {
    return { x: this.fx.filter(x, t), y: this.fy.filter(y, t) };
  }

  last(): { x: number; y: number } | null {
    const x = this.fx.lastValue();
    const y = this.fy.lastValue();
    return x === null || y === null ? null : { x, y };
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }
}
