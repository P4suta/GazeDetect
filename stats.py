"""セッション統計（純粋ロジック、cv2 非依存）。

維持率の分母は「壁時計時間」ではなく「顔が映っていた時間」。席を立つなど顔が
フレームから外れている間はタイマーを止め、目線維持率が汚れないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gaze import ContactState

# 逸れ方向コード → 日本語ラベル
_DRIFT_LABELS = {
    "down": "下（画面・手元）",
    "up": "上",
    "left": "左",
    "right": "右",
}


@dataclass
class SessionStats:
    """1 セッション分の目線維持の集計。"""

    total_time: float = 0.0  # 経過時間（全フレーム）
    visible_time: float = 0.0  # 顔が映っていた時間（維持率の分母）
    contact_time: float = 0.0  # カメラ目線だった時間
    current_streak: float = 0.0  # 現在の連続キープ時間
    longest_streak: float = 0.0  # 最長キープ時間
    drift_time: dict[str, float] = field(
        default_factory=lambda: {"down": 0.0, "up": 0.0, "left": 0.0, "right": 0.0}
    )

    def update(
        self,
        dt: float,
        state: ContactState | None,
        drift: str | None = None,
    ) -> None:
        """1 フレーム分を反映する。state=None は顔ロスト（タイマー停止）。"""
        self.total_time += dt
        if state is None:
            self.current_streak = 0.0
            return
        self.visible_time += dt
        if state is ContactState.CONTACT:
            self.contact_time += dt
            self.current_streak += dt
            self.longest_streak = max(self.longest_streak, self.current_streak)
        else:
            self.current_streak = 0.0
            if drift is not None and drift in self.drift_time:
                self.drift_time[drift] += dt

    @property
    def contact_ratio(self) -> float:
        """顔可視時間に対するカメラ目線の割合（0.0–1.0）。"""
        return self.contact_time / self.visible_time if self.visible_time > 0 else 0.0

    def dominant_drift(self) -> str | None:
        """最も長く視線が外れていた向き（無ければ None）。"""
        if not any(self.drift_time.values()):
            return None
        return max(self.drift_time, key=lambda k: self.drift_time[k])

    def reset(self) -> None:
        self.total_time = 0.0
        self.visible_time = 0.0
        self.contact_time = 0.0
        self.current_streak = 0.0
        self.longest_streak = 0.0
        for key in self.drift_time:
            self.drift_time[key] = 0.0

    def summary_lines(self) -> list[str]:
        """終了時にターミナルへ出すサマリー。"""
        lines = [
            "===== セッション結果 =====",
            f"経過時間          : {self.total_time:6.1f} 秒",
            f"顔が映っていた時間: {self.visible_time:6.1f} 秒",
            f"カメラ目線維持率  : {self.contact_ratio * 100:5.1f} %",
            f"最長キープ        : {self.longest_streak:6.1f} 秒",
        ]
        dominant = self.dominant_drift()
        if dominant is not None:
            lines.append(f"逸れがちな向き    : {_DRIFT_LABELS.get(dominant, dominant)}")
        lines.append("==========================")
        return lines
