"""画面描画（OpenCV + Pillow）。

呼び出し側は「表示用（左右反転済み）」フレームを渡す。OpenCV の ``putText`` は
日本語を描けないため、テキストは Pillow で 1 フレーム 1 パスだけまとめて描画する
（Pillow は mediapipe が依存として導入済み）。日本語フォントが見つからない環境では
ASCII の既定フォントへフォールバックする。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gaze import (
    LEFT_EYE_CORNERS,
    LEFT_IRIS,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS,
    ContactState,
    GazeDebug,
)
from stats import SessionStats

# 色は OpenCV 慣習の BGR。
GREEN = (0, 200, 0)
RED = (0, 0, 220)
GRAY = (150, 150, 150)
WHITE = (240, 240, 240)
YELLOW = (0, 215, 255)
CYAN = (255, 200, 0)

_MARGIN = 20

# Windows 標準の日本語フォント候補。
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
)

# 描画テキスト 1 件: (本文, 左上座標, 色BGR, フォントサイズ)
TextItem = tuple[str, tuple[int, int], tuple[int, int, int], int]

# Pillow のフォント型（TrueType と既定フォントの双方を許容）。
FontT = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass
class OverlayState:
    """1 フレーム分の UI 状態。"""

    mode: str  # "WAIT_FOR_FACE" | "CALIBRATING" | "ACTIVE"
    debug: GazeDebug | None = None
    face_visible: bool = False
    calib_progress: float = 0.0  # 0.0–1.0
    calib_remaining: float = 0.0  # 残り秒
    calib_count: int = 0  # キャリブで採取できた有効フレーム数
    calib_min: int = 0  # 必要フレーム数
    show_debug: bool = False
    fps: float = 0.0
    landmarks_mirrored: np.ndarray | None = None  # 表示座標系の 478×2


class OverlayRenderer:
    """フォントをキャッシュし、1 フレームの描画を 1 パスで行う。"""

    def __init__(self) -> None:
        self._font_path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        self._fonts: dict[int, FontT] = {}

    def _font(self, size: int) -> FontT:
        if size not in self._fonts:
            if self._font_path is not None:
                self._fonts[size] = ImageFont.truetype(self._font_path, size)
            else:
                self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]

    def _draw_texts(self, frame: np.ndarray, items: list[TextItem]) -> np.ndarray:
        """全テキストを Pillow で 1 パス描画し、BGR フレームを返す。"""
        img = Image.fromarray(frame[:, :, ::-1].copy())
        draw = ImageDraw.Draw(img)
        for text, (x, y), color_bgr, size in items:
            rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            font = self._font(size)
            draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))  # 影
            draw.text((x, y), text, font=font, fill=rgb)
        return np.asarray(img)[:, :, ::-1].copy()

    def render(self, frame: np.ndarray, state: OverlayState, stats: SessionStats) -> np.ndarray:
        height, width = frame.shape[:2]

        border = self._border_color(state)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border, 12)

        if state.landmarks_mirrored is not None:
            self._draw_landmarks(frame, state.landmarks_mirrored)
        if state.mode == "CALIBRATING":
            self._draw_calib_target(frame, state.calib_progress)

        items = self._build_texts(state, stats, width, height)
        return self._draw_texts(frame, items)

    # --- 内部ヘルパ ---

    @staticmethod
    def _border_color(state: OverlayState) -> tuple[int, int, int]:
        if state.mode == "CALIBRATING":
            return YELLOW
        if state.mode == "ACTIVE" and state.face_visible and state.debug is not None:
            return GREEN if state.debug.state is ContactState.CONTACT else RED
        return GRAY

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, pts: np.ndarray) -> None:
        for ring in (LEFT_IRIS, RIGHT_IRIS):
            center = pts[list(ring)].mean(axis=0)
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, GREEN, -1)
        for idx in (*LEFT_EYE_CORNERS, *RIGHT_EYE_CORNERS):
            cv2.circle(frame, (int(pts[idx, 0]), int(pts[idx, 1])), 2, CYAN, -1)

    @staticmethod
    def _draw_calib_target(frame: np.ndarray, progress: float) -> None:
        width = frame.shape[1]
        target = (width // 2, 34)
        cv2.circle(frame, target, 10, GREEN, -1)
        cv2.circle(frame, target, 16, WHITE, 2)
        cv2.ellipse(frame, target, (24, 24), -90, 0, int(360 * progress), YELLOW, 3)

    def _build_texts(
        self, state: OverlayState, stats: SessionStats, width: int, height: int
    ) -> list[TextItem]:
        items: list[TextItem] = []
        x = _MARGIN
        y = _MARGIN

        # ステータス（大）
        status_text, status_color = self._status_line(state)
        items.append((status_text, (x, y), status_color, 30))
        y += 46

        # HUD（維持率・キープ）
        hud = (
            f"維持率 {stats.contact_ratio * 100:4.0f}%   "
            f"最長 {stats.longest_streak:4.1f}s   "
            f"現在 {stats.current_streak:4.1f}s"
        )
        items.append((hud, (x, y), WHITE, 22))
        y += 34

        # デバッグ数値（ASCII）
        if state.show_debug and state.debug is not None:
            d = state.debug
            z = d.zscores
            for line in (
                f"dev {d.smoothed:.2f} (raw {d.deviation:.2f})  fps {state.fps:.1f}",
                f"z: iris_v {z[1]:+.2f}  pitch {z[4]:+.2f}  iris_h {z[0]:+.2f}  yaw {z[3]:+.2f}",
                f"ear {d.metrics.ear:.3f}  yaw {d.metrics.yaw:+.3f}  pitch {d.metrics.pitch:+.3f}"
                + ("  [BLINK]" if d.blink else "")
                + ("  [YAW-GATE]" if d.yaw_gated else ""),
            ):
                items.append((line, (x, y), YELLOW, 18))
                y += 26

        # 操作ヒント（下部）
        help_text = "q:終了   c:再キャリブ   d:デバッグ   r:リセット"
        items.append((help_text, (x, height - 34), GRAY, 18))
        return items

    @staticmethod
    def _status_line(state: OverlayState) -> tuple[str, tuple[int, int, int]]:
        if state.mode == "WAIT_FOR_FACE":
            return "顔を画面に入れてください", GRAY
        if state.mode == "CALIBRATING":
            return (
                f"キャリブレーション中… カメラのレンズを見て"
                f"（残り{state.calib_remaining:.0f}秒  {state.calib_count}/{state.calib_min}）",
                YELLOW,
            )
        # ACTIVE
        if not state.face_visible or state.debug is None:
            return "顔を見失いました", GRAY
        if state.debug.state is ContactState.CONTACT:
            return "◎ カメラ目線 OK", GREEN
        return "✕ 目線をカメラへ", RED
