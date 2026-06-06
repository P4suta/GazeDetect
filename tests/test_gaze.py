"""純粋ロジック層のテスト（cv2 / mediapipe 不要）。

合成ランドマークで虹彩比率・キャリブレーション・ヒステリシス分類を検証する。
"""

from __future__ import annotations

import numpy as np

import gaze
from gaze import (
    Calibrator,
    ClassifierConfig,
    ContactState,
    GazeClassifier,
    GazeMetrics,
    compute_metrics,
    drift_direction,
)

_EW = 40.0  # 目の幅
_EH = 14.0  # 目の高さ
_R = 5.0  # 虹彩リング半径


def _set_eye(
    pts: np.ndarray,
    cx: float,
    cy: float,
    corners: tuple[int, int],
    lids: tuple[int, int],
    iris: tuple[int, ...],
    ear_vert: tuple[tuple[int, int], ...],
    iris_dx: float,
    iris_dy: float,
) -> None:
    pts[corners[0]] = [cx - _EW / 2, cy]
    pts[corners[1]] = [cx + _EW / 2, cy]
    pts[lids[0]] = [cx, cy - _EH / 2]
    pts[lids[1]] = [cx, cy + _EH / 2]
    for top, bottom in ear_vert:
        pts[top] = [cx, cy - _EH / 2]
        pts[bottom] = [cx, cy + _EH / 2]
    icx, icy = cx + iris_dx, cy + iris_dy
    ring = list(iris)
    pts[ring[0]] = [icx, icy]
    pts[ring[1]] = [icx + _R, icy]
    pts[ring[2]] = [icx - _R, icy]
    pts[ring[3]] = [icx, icy + _R]
    pts[ring[4]] = [icx, icy - _R]


def make_landmarks(iris_dx: float = 0.0, iris_dy: float = 0.0) -> np.ndarray:
    """正面・両目に同一の虹彩オフセットを与えた合成ランドマークを作る。"""
    pts = np.zeros((478, 2), dtype=np.float64)
    pts[gaze.FOREHEAD] = [320, 120]
    pts[gaze.CHIN] = [320, 400]
    pts[gaze.NOSE_TIP] = [320, 270]
    pts[gaze.LEFT_CHEEK] = [220, 260]
    pts[gaze.RIGHT_CHEEK] = [420, 260]
    _set_eye(
        pts,
        260,
        220,
        gaze.LEFT_EYE_CORNERS,
        gaze.LEFT_EYE_LIDS,
        gaze.LEFT_IRIS,
        gaze.LEFT_EAR_VERT,
        iris_dx,
        iris_dy,
    )
    _set_eye(
        pts,
        380,
        220,
        gaze.RIGHT_EYE_CORNERS,
        gaze.RIGHT_EYE_LIDS,
        gaze.RIGHT_IRIS,
        gaze.RIGHT_EAR_VERT,
        iris_dx,
        iris_dy,
    )
    return pts


def test_neutral_metrics_are_centered() -> None:
    m = compute_metrics(make_landmarks())
    assert m.iris_h == 0.5
    assert m.iris_v == 0.5
    assert m.ear == 0.35  # _EH / _EW


def test_iris_down_increases_vertical_ratio() -> None:
    neutral = compute_metrics(make_landmarks())
    down = compute_metrics(make_landmarks(iris_dy=4.0))
    assert down.iris_v > neutral.iris_v


def test_uniform_horizontal_shift_does_not_cancel() -> None:
    """画像座標 min/max 方式なら、両目同方向の虹彩移動が平均で打ち消されない。"""
    neutral = compute_metrics(make_landmarks())
    shifted = compute_metrics(make_landmarks(iris_dx=6.0))
    assert shifted.iris_h > neutral.iris_h + 0.1


def test_mirror_reflects_horizontal_ratio_about_half() -> None:
    width = 640
    pts = make_landmarks(iris_dx=6.0)
    mirrored = pts.copy()
    mirrored[:, 0] = (width - 1) - mirrored[:, 0]
    original = compute_metrics(pts)
    flipped = compute_metrics(mirrored)
    assert flipped.iris_h == 1.0 - original.iris_h


def _calibration_from_neutral(samples: int = 40) -> gaze.Calibration:
    calibrator = Calibrator()
    rng = np.random.default_rng(0)
    for _ in range(samples):
        jitter = float(rng.normal(0.0, 0.3))
        calibrator.add(compute_metrics(make_landmarks(iris_dx=jitter, iris_dy=jitter)))
    return calibrator.finish()


def test_calibrator_rejects_closed_eyes() -> None:
    calibrator = Calibrator(ear_floor=0.12)
    closed = GazeMetrics(iris_h=0.5, iris_v=0.5, ear=0.05, yaw=0.0, pitch=0.1)
    assert calibrator.add(closed) is False
    assert calibrator.count == 0


def test_zscore_large_for_downward_gaze() -> None:
    calib = _calibration_from_neutral()
    z = calib.zscores(compute_metrics(make_landmarks(iris_dy=4.0)))
    assert z[1] > 3.0  # iris_v の z スコアが大きい


def test_classifier_enters_and_exits_contact() -> None:
    calib = _calibration_from_neutral()
    clf = GazeClassifier(calib, ClassifierConfig())
    neutral = compute_metrics(make_landmarks())

    for _ in range(10):  # 正面注視を継続 → CONTACT へ
        clf.update(neutral, dt=0.05)
    assert clf.state is ContactState.CONTACT

    down = compute_metrics(make_landmarks(iris_dy=5.0))
    for _ in range(10):  # 下を見続ける → NO_CONTACT へ
        clf.update(down, dt=0.05)
    assert clf.state is ContactState.NO_CONTACT


def test_blink_freezes_state() -> None:
    calib = _calibration_from_neutral()
    clf = GazeClassifier(calib)
    neutral = compute_metrics(make_landmarks())
    for _ in range(10):
        clf.update(neutral, dt=0.05)
    assert clf.state is ContactState.CONTACT

    blink = GazeMetrics(iris_h=0.5, iris_v=0.9, ear=0.05, yaw=0.0, pitch=0.18)
    result = clf.update(blink, dt=0.05)
    assert result.blink is True
    assert clf.state is ContactState.CONTACT  # 瞬きで NG に倒れない


def test_yaw_gate_forces_no_contact() -> None:
    calib = _calibration_from_neutral()
    clf = GazeClassifier(calib)
    neutral = compute_metrics(make_landmarks())
    for _ in range(10):
        clf.update(neutral, dt=0.05)
    assert clf.state is ContactState.CONTACT

    turned = GazeMetrics(iris_h=0.5, iris_v=0.5, ear=0.35, yaw=0.5, pitch=0.18)
    result = clf.update(turned, dt=0.05)
    assert result.yaw_gated is True
    assert clf.state is ContactState.NO_CONTACT


def test_drift_direction() -> None:
    assert drift_direction(np.array([0.0, 5.0, 0.0, 0.0, 0.0])) == "down"
    assert drift_direction(np.array([0.0, -5.0, 0.0, 0.0, 0.0])) == "up"
    assert drift_direction(np.array([5.0, 0.0, 0.0, 0.0, 0.0])) == "right"
    assert drift_direction(np.array([-5.0, 0.0, 0.0, 0.0, 0.0])) == "left"
    assert drift_direction(np.array([0.2, 0.2, 0.0, 0.0, 0.0])) == "center"
