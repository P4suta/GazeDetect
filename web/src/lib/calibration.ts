// 多点キャリブレーション（純粋）。固視ゲートで採取した品質サンプル（重み付き）を外れ値除去し、
// 品質重み付きリッジ回帰で「特徴 → 画面 PoR」を学習して GazeModel を作る。

import { GazeModel } from "./gaze";
import { applyWeights, ridgeFitWeighted, Standardizer } from "./linalg";

export interface CalibTarget {
  id: string;
  x: number;
  y: number;
  isCamera: boolean;
  label?: string;
}

// camPoint が実測できない場合のフォールバック（画面上端中央の少し上）。
export const CAMERA_POINT = { x: 0.5, y: -0.12 };
export const CAMERA_TARGET_ID = "camera";

function buildTargets(): CalibTarget[] {
  const grid = [0.12, 0.5, 0.88];
  const targets: CalibTarget[] = [];
  for (const gy of grid) {
    for (const gx of grid) {
      targets.push({ id: `g-${gx}-${gy}`, x: gx, y: gy, isCamera: false });
    }
  }
  // カメラ点は「レンズ注視」の特徴採取用。座標は学習には使わず（後で実測）、表示の手掛かりのみ。
  targets.push({
    id: CAMERA_TARGET_ID,
    x: CAMERA_POINT.x,
    y: CAMERA_POINT.y,
    isCamera: true,
    label: "カメラのレンズ",
  });
  return targets;
}

export const CALIB_TARGETS: CalibTarget[] = buildTargets();
// 学習に使う画面グリッド点（座標が既知の点のみ）。カメラ点は学習に含めない。
export const SCREEN_TARGETS: CalibTarget[] = CALIB_TARGETS.filter((t) => !t.isCamera);

interface SampleRec {
  f: number[];
  w: number;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export class CalibrationCollector {
  private samples = new Map<string, SampleRec[]>();

  // weight は固視の安定度などの品質（高いほど信頼）。
  add(targetId: string, features: number[], weight = 1): void {
    const arr = this.samples.get(targetId) ?? [];
    arr.push({ f: features, w: weight });
    this.samples.set(targetId, arr);
  }

  count(targetId: string): number {
    return this.samples.get(targetId)?.length ?? 0;
  }

  reset(): void {
    this.samples.clear();
  }

  // 各次元の中央値から MAD の 4 倍を超えるサンプルを外れ値として除去。
  private cleaned(recs: SampleRec[]): SampleRec[] {
    if (recs.length < 4) {
      return recs;
    }
    const d = recs[0].f.length;
    const med = new Array<number>(d);
    const mad = new Array<number>(d);
    for (let i = 0; i < d; i++) {
      med[i] = median(recs.map((r) => r.f[i]));
    }
    for (let i = 0; i < d; i++) {
      mad[i] = median(recs.map((r) => Math.abs(r.f[i] - med[i]))) || 1e-6;
    }
    return recs.filter((r) => r.f.every((v, i) => Math.abs(v - med[i]) <= 4 * mad[i]));
  }

  // クリーン済みサンプルの平均特徴（外挿で camPoint を実測するのに使う）。
  meanFeatures(targetId: string): number[] | null {
    const recs = this.cleaned(this.samples.get(targetId) ?? []);
    if (recs.length === 0) {
      return null;
    }
    const d = recs[0].f.length;
    const mean = new Array<number>(d).fill(0);
    for (const r of recs) {
      for (let i = 0; i < d; i++) {
        mean[i] += r.f[i];
      }
    }
    return mean.map((v) => v / recs.length);
  }

  // 座標が既知の trainTargets だけで回帰を学習。camPoint は cameraTargetId のレンズ注視特徴を
  // 学習済みモデルで予測して「実測」する（ハードコードしない）。採取がなければフォールバック。
  fit(trainTargets: CalibTarget[], lambda = 1.0, cameraTargetId?: string): GazeModel {
    const X: number[][] = [];
    const Y: number[][] = [];
    const W: number[] = [];
    for (const t of trainTargets) {
      for (const r of this.cleaned(this.samples.get(t.id) ?? [])) {
        X.push(r.f);
        Y.push([t.x, t.y]);
        W.push(r.w);
      }
    }
    if (X.length < trainTargets.length) {
      throw new Error("キャリブレーションのサンプルが不足しています");
    }
    const scaler = Standardizer.fit(X);
    const xs = X.map((r) => [1, ...scaler.transform(r)]);
    const weights = ridgeFitWeighted(xs, Y, W, lambda);

    let camPoint = { ...CAMERA_POINT };
    const camMean = cameraTargetId ? this.meanFeatures(cameraTargetId) : null;
    if (camMean) {
      const [px, py] = applyWeights(weights, [1, ...scaler.transform(camMean)]);
      camPoint = { x: px, y: py };
    }
    return new GazeModel(scaler, weights, camPoint);
  }
}
