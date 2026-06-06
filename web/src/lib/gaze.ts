// 視線推定の判定ロジック（純粋・DOM 非依存）。
// 特徴 → リッジ回帰で画面上の注視点(PoR) → One-Euro 平滑化 → カメラ点との距離で
// 「カメラ目線」を二重しきい値ヒステリシス判定。瞬きは凍結。

import { PointFilter } from "./filter";
import { applyWeights, type Matrix, type Standardizer } from "./linalg";

export type GazePoint = { x: number; y: number };

export const ContactState = {
  Contact: "contact",
  NoContact: "no_contact",
} as const;
export type ContactState = (typeof ContactState)[keyof typeof ContactState];

function dist(a: GazePoint, b: GazePoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// キャリブで学習した「特徴 → 画面 PoR」写像。
export class GazeModel {
  constructor(
    readonly scaler: Standardizer,
    readonly weights: Matrix, // (FEATURE_DIM+1)×2、先頭行はバイアス
    readonly camPoint: GazePoint, // カメラ位置（画面正規化座標）
  ) {}

  predict(rawFeatures: number[]): GazePoint {
    const x = [1, ...this.scaler.transform(rawFeatures)];
    const [px, py] = applyWeights(this.weights, x);
    return { x: px, y: py };
  }
}

export interface EyeContactConfig {
  enterR: number; // この距離未満が継続で CONTACT 進入
  exitR: number; // この距離超が継続で CONTACT 退出
  enterMs: number;
  exitMs: number;
  blinkThresh: number; // eyeBlink がこれ超で瞬き＝判定凍結
  minCutoff: number; // One-Euro
  beta: number; // One-Euro
}

export const DEFAULT_EC_CONFIG: EyeContactConfig = {
  enterR: 0.13,
  exitR: 0.22,
  enterMs: 120,
  exitMs: 200,
  blinkThresh: 0.5,
  minCutoff: 1.2,
  beta: 0.6,
};

export interface EyeContactDebug {
  por: GazePoint; // 平滑後
  rawPor: GazePoint; // 平滑前
  distance: number; // カメラ点との距離（平滑後）
  state: ContactState;
  blink: boolean;
}

export class EyeContactClassifier {
  private filter: PointFilter;
  private state: ContactState = ContactState.NoContact;
  private enterElapsed = 0;
  private exitElapsed = 0;
  private t = 0;

  constructor(
    readonly model: GazeModel,
    readonly config: EyeContactConfig = DEFAULT_EC_CONFIG,
  ) {
    this.filter = new PointFilter(config.minCutoff, config.beta);
  }

  get currentState(): ContactState {
    return this.state;
  }

  update(rawFeatures: number[], blink: number, dt: number): EyeContactDebug {
    this.t += dt;
    const raw = this.model.predict(rawFeatures);

    // 瞬き中は平滑値・状態を凍結
    if (blink > this.config.blinkThresh) {
      const held = this.filter.last() ?? raw;
      return {
        por: held,
        rawPor: raw,
        distance: dist(held, this.model.camPoint),
        state: this.state,
        blink: true,
      };
    }

    const por = this.filter.filter(raw.x, raw.y, this.t);
    const distance = dist(por, this.model.camPoint);
    this.applyHysteresis(distance, dt);
    return { por, rawPor: raw, distance, state: this.state, blink: false };
  }

  private applyHysteresis(distance: number, dt: number): void {
    const c = this.config;
    if (this.state === ContactState.NoContact) {
      if (distance < c.enterR) {
        this.enterElapsed += dt * 1000;
        if (this.enterElapsed >= c.enterMs) {
          this.state = ContactState.Contact;
          this.exitElapsed = 0;
        }
      } else {
        this.enterElapsed = 0;
      }
    } else if (distance > c.exitR) {
      this.exitElapsed += dt * 1000;
      if (this.exitElapsed >= c.exitMs) {
        this.state = ContactState.NoContact;
        this.enterElapsed = 0;
      }
    } else {
      this.exitElapsed = 0;
    }
  }

  reset(): void {
    this.filter.reset();
    this.state = ContactState.NoContact;
    this.enterElapsed = 0;
    this.exitElapsed = 0;
    this.t = 0;
  }
}

// 逸れた向き（統計用）。PoR がカメラ点からどちらにずれているか。
export function driftDirection(por: GazePoint, cam: GazePoint, threshold = 0.1): string {
  const dx = por.x - cam.x;
  const dy = por.y - cam.y;
  if (Math.hypot(dx, dy) < threshold) {
    return "center";
  }
  if (Math.abs(dy) >= Math.abs(dx)) {
    return dy > 0 ? "down" : "up";
  }
  return dx > 0 ? "right" : "left";
}
