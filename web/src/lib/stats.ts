// セッション統計（純粋・DOM 非依存）。stats.py からの移植。
// 維持率の分母は「顔が映っていた時間」（壁時計ではない）。顔ロスト中はタイマーを止める。

import { ContactState } from "./gaze";

// 逸れ方向コード → 日本語ラベル
export const DRIFT_LABELS: Record<string, string> = {
  down: "下（画面・手元）",
  up: "上",
  left: "左",
  right: "右",
};

export class SessionStats {
  totalTime = 0; // 経過時間（全フレーム）
  visibleTime = 0; // 顔が映っていた時間（維持率の分母）
  contactTime = 0; // カメラ目線だった時間
  currentStreak = 0; // 現在の連続キープ時間
  longestStreak = 0; // 最長キープ時間
  driftTime: Record<string, number> = { down: 0, up: 0, left: 0, right: 0 };

  // 1 フレーム分を反映する。state=null は顔ロスト（タイマー停止）。
  update(dt: number, state: ContactState | null, drift: string | null = null): void {
    this.totalTime += dt;
    if (state === null) {
      this.currentStreak = 0;
      return;
    }
    this.visibleTime += dt;
    if (state === ContactState.Contact) {
      this.contactTime += dt;
      this.currentStreak += dt;
      this.longestStreak = Math.max(this.longestStreak, this.currentStreak);
    } else {
      this.currentStreak = 0;
      if (drift !== null && Object.hasOwn(this.driftTime, drift)) {
        this.driftTime[drift] += dt;
      }
    }
  }

  get contactRatio(): number {
    return this.visibleTime > 0 ? this.contactTime / this.visibleTime : 0;
  }

  dominantDrift(): string | null {
    let best: string | null = null;
    let bestValue = 0;
    for (const [key, value] of Object.entries(this.driftTime)) {
      if (value > bestValue) {
        bestValue = value;
        best = key;
      }
    }
    return best;
  }

  reset(): void {
    this.totalTime = 0;
    this.visibleTime = 0;
    this.contactTime = 0;
    this.currentStreak = 0;
    this.longestStreak = 0;
    this.driftTime = { down: 0, up: 0, left: 0, right: 0 };
  }

  summaryLines(): string[] {
    const lines = [
      `経過時間          : ${this.totalTime.toFixed(1)} 秒`,
      `顔が映っていた時間: ${this.visibleTime.toFixed(1)} 秒`,
      `カメラ目線維持率  : ${(this.contactRatio * 100).toFixed(1)} %`,
      `最長キープ        : ${this.longestStreak.toFixed(1)} 秒`,
    ];
    const dominant = this.dominantDrift();
    if (dominant !== null) {
      lines.push(`逸れがちな向き    : ${DRIFT_LABELS[dominant] ?? dominant}`);
    }
    return lines;
  }
}
