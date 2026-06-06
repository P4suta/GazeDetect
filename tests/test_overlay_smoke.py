"""overlay の実行時スモークテスト（ディスプレイ不要）。

cv2 の図形描画と Pillow の日本語テキスト描画が実機のフォントで例外なく動き、
入力と同じ形状・dtype のフレームを返すことを確認する（imshow は呼ばない）。
"""

from __future__ import annotations

import numpy as np

from gaze import ContactState, GazeDebug, GazeMetrics
from overlay import OverlayRenderer, OverlayState
from stats import SessionStats

_H, _W = 480, 640


def _debug(state: ContactState) -> GazeDebug:
    return GazeDebug(
        metrics=GazeMetrics(iris_h=0.5, iris_v=0.5, ear=0.3, yaw=0.0, pitch=0.18),
        zscores=np.zeros(5, dtype=np.float64),
        deviation=0.4,
        smoothed=0.4,
        state=state,
        yaw_gated=False,
    )


def _blank() -> np.ndarray:
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def _assert_valid(frame: np.ndarray) -> None:
    assert frame.shape == (_H, _W, 3)
    assert frame.dtype == np.uint8


def test_render_each_mode() -> None:
    renderer = OverlayRenderer()
    stats = SessionStats()
    stats.update(1.0, ContactState.CONTACT)

    states = [
        OverlayState(mode="WAIT_FOR_FACE"),
        OverlayState(mode="CALIBRATING", calib_progress=0.5, calib_remaining=1.5),
        OverlayState(mode="ACTIVE", debug=_debug(ContactState.CONTACT), face_visible=True),
        OverlayState(mode="ACTIVE", debug=_debug(ContactState.NO_CONTACT), face_visible=True),
        OverlayState(mode="ACTIVE", face_visible=False),  # 顔ロスト
    ]
    for state in states:
        _assert_valid(renderer.render(_blank(), state, stats))


def test_render_with_debug_and_landmarks() -> None:
    renderer = OverlayRenderer()
    stats = SessionStats()
    pts = np.full((478, 2), 100.0, dtype=np.float64)
    state = OverlayState(
        mode="ACTIVE",
        debug=_debug(ContactState.CONTACT),
        face_visible=True,
        show_debug=True,
        fps=30.0,
        landmarks_mirrored=pts,
    )
    _assert_valid(renderer.render(_blank(), state, stats))
