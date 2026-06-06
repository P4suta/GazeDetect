// キャリブレーションの中核ロジック（純粋・DOM 非依存）。
// 「OK は演出ではなく勝ち取る」: 各点は、適応しきい値以下の固視を“連続して”必要時間保持できた
// ときだけ lock する。崩れたら猶予つきで減衰（瞬きで台無しにしない）。学習後に held-out 点で
// 精度を検証する。

import {
  CALIB_TARGETS,
  CAMERA_TARGET_ID,
  CalibrationCollector,
  SCREEN_TARGETS,
} from "./calibration";
import { AdaptiveThreshold, FixationDetector } from "./fixation";
import type { GazeModel, GazePoint } from "./gaze";

// 学習に使わない検証点（9 グリッドと重ならない位置）。
export const VALIDATION_TARGETS: GazePoint[] = [
  { x: 0.3, y: 0.32 },
  { x: 0.7, y: 0.32 },
  { x: 0.5, y: 0.72 },
];

export interface EngineConfig {
  settle: number; // 点を見つける整定(秒)
  hold: number; // ロックに必要な「連続良好固視」時間(秒)
  confirm: number; // ロック後の確定表示(秒)
  grace: number; // 崩れても減衰しない猶予(秒)（瞬き対策）
  drain: number; // 崩れ時の hold 減衰倍率
  blinkThresh: number;
  percentile: number; // 適応しきい値のパーセンタイル
  lambda: number; // リッジ正則化
}

export const DEFAULT_ENGINE_CONFIG: EngineConfig = {
  settle: 0.5,
  hold: 1.0,
  confirm: 0.45,
  grace: 0.15,
  drain: 1.5,
  blinkThresh: 0.5,
  percentile: 0.55,
  lambda: 1.0,
};

export type Phase =
  | "settle"
  | "acquire"
  | "confirm"
  | "validate-settle"
  | "validate-acquire"
  | "done";

export interface DisplayTarget {
  x: number;
  y: number;
  label: string;
  isCamera: boolean;
  kind: "calibrate" | "validate";
}

export interface FrameInput {
  features: number[];
  gaze: GazePoint;
  blink: number;
}

export interface EngineState {
  phase: Phase;
  target: DisplayTarget | null;
  index: number; // 現フェーズ内の点番号(0始まり)
  total: number;
  holdProgress: number; // 0..1
  dispersion: number;
  threshold: number;
  good: boolean;
  faceVisible: boolean;
  justLocked: boolean;
  done: boolean;
  model: GazeModel | null;
  accuracy: number | null; // held-out 平均誤差（画面正規化、低いほど良い）
}

function dist(a: GazePoint, b: GazePoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export class CalibrationEngine {
  private readonly cfg: EngineConfig;
  private readonly collector = new CalibrationCollector();
  private readonly fix = new FixationDetector(0.2, Number.POSITIVE_INFINITY);
  private readonly adapt: AdaptiveThreshold;

  private phase: Phase = "settle";
  private idx = 0;
  private phaseT = 0;
  private hold = 0;
  private badT = 0;
  private model: GazeModel | null = null;

  private valIdx = 0;
  private valAccum: GazePoint = { x: 0, y: 0 };
  private valCount = 0;
  private valErrors: number[] = [];
  private accuracy: number | null = null;

  constructor(config: Partial<EngineConfig> = {}) {
    this.cfg = { ...DEFAULT_ENGINE_CONFIG, ...config };
    this.adapt = new AdaptiveThreshold(6, this.cfg.percentile, 20);
  }

  private currentTarget(): DisplayTarget | null {
    if (this.phase === "settle" || this.phase === "acquire" || this.phase === "confirm") {
      const t = CALIB_TARGETS[this.idx];
      return {
        x: t.x,
        y: t.y,
        label: t.isCamera ? "カメラのレンズ" : "光る点",
        isCamera: t.isCamera,
        kind: "calibrate",
      };
    }
    if (this.phase === "validate-settle" || this.phase === "validate-acquire") {
      const v = VALIDATION_TARGETS[this.valIdx];
      return { x: v.x, y: v.y, label: "確認の点", isCamera: false, kind: "validate" };
    }
    return null;
  }

  feed(input: FrameInput | null, dt: number, t: number): EngineState {
    const valid = input !== null && input.blink < this.cfg.blinkThresh;
    let disp = Number.POSITIVE_INFINITY;
    if (valid && input) {
      this.fix.push(input.gaze.x, input.gaze.y, t);
      disp = this.fix.dispersion();
      this.adapt.push(disp, t);
    }
    const threshold = this.adapt.value();
    const good = valid && disp <= threshold;

    this.phaseT += dt;
    let justLocked = false;

    switch (this.phase) {
      case "settle":
        if (this.phaseT >= this.cfg.settle) {
          this.enterAcquire("acquire");
        }
        break;
      case "acquire":
        this.acquireStep(input, good, dt, true);
        if (this.hold >= this.cfg.hold) {
          justLocked = true;
          this.phase = "confirm";
          this.phaseT = 0;
        }
        break;
      case "confirm":
        if (this.phaseT >= this.cfg.confirm) {
          this.advanceCalibrate();
        }
        break;
      case "validate-settle":
        if (this.phaseT >= this.cfg.settle) {
          this.enterAcquire("validate-acquire");
          this.valAccum = { x: 0, y: 0 };
          this.valCount = 0;
        }
        break;
      case "validate-acquire":
        this.acquireStep(input, good, dt, false);
        if (this.hold >= this.cfg.hold) {
          justLocked = true;
          this.lockValidate();
        }
        break;
      case "done":
        break;
    }

    return {
      phase: this.phase,
      target: this.currentTarget(),
      index: this.phase.startsWith("validate") ? this.valIdx : this.idx,
      total: this.phase.startsWith("validate") ? VALIDATION_TARGETS.length : CALIB_TARGETS.length,
      holdProgress: Math.min(this.hold / this.cfg.hold, 1),
      dispersion: disp,
      threshold,
      good,
      faceVisible: input !== null,
      justLocked,
      done: this.phase === "done",
      model: this.model,
      accuracy: this.accuracy,
    };
  }

  private enterAcquire(next: Phase): void {
    this.phase = next;
    this.phaseT = 0;
    this.hold = 0;
    this.badT = 0;
    this.fix.reset();
  }

  private acquireStep(
    input: FrameInput | null,
    good: boolean,
    dt: number,
    calibrate: boolean,
  ): void {
    if (good && input) {
      this.hold += dt;
      this.badT = 0;
      if (calibrate) {
        const target = CALIB_TARGETS[this.idx];
        const q = Math.min(1, Math.max(0.2, 1 - this.fix.dispersion() / this.adapt.value()));
        this.collector.add(target.id, input.features, Number.isFinite(q) ? q : 1);
      } else if (this.model) {
        const p = this.model.predict(input.features);
        this.valAccum = { x: this.valAccum.x + p.x, y: this.valAccum.y + p.y };
        this.valCount += 1;
      }
    } else {
      this.badT += dt;
      if (this.badT > this.cfg.grace) {
        this.hold = Math.max(0, this.hold - this.cfg.drain * dt);
      }
    }
  }

  private advanceCalibrate(): void {
    this.idx += 1;
    if (this.idx < CALIB_TARGETS.length) {
      this.phase = "settle";
      this.phaseT = 0;
    } else {
      // 学習は座標既知のグリッドのみ。camPoint はレンズ注視特徴の外挿で実測する。
      this.model = this.collector.fit(SCREEN_TARGETS, this.cfg.lambda, CAMERA_TARGET_ID);
      this.valIdx = 0;
      this.phase = "validate-settle";
      this.phaseT = 0;
    }
  }

  private lockValidate(): void {
    const mean =
      this.valCount > 0
        ? { x: this.valAccum.x / this.valCount, y: this.valAccum.y / this.valCount }
        : { x: 0, y: 0 };
    this.valErrors.push(dist(mean, VALIDATION_TARGETS[this.valIdx]));
    this.valIdx += 1;
    if (this.valIdx < VALIDATION_TARGETS.length) {
      this.phase = "validate-settle";
      this.phaseT = 0;
    } else {
      this.accuracy = this.valErrors.reduce((a, b) => a + b, 0) / this.valErrors.length;
      this.phase = "done";
    }
  }
}
