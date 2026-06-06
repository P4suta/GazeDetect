"""カメラ目線（アイコンタクト）検知のコアロジック。

このモジュールは 2 層で構成される:

1. 純粋ロジック層 — numpy にのみ依存し、cv2 / mediapipe を import しない。
   478 点のランドマーク配列を入力に、虹彩比率・EAR・頭部プロキシ・キャリブレーション・
   ヒステリシス付き分類を行う。合成データで単体テストできる。
2. ``FaceMeshTracker`` — mediapipe を「遅延 import」する薄いラッパ。RGB フレームから
   478 点のランドマーク（ピクセル座標）を返す。

座標系の約束: 推論・キャリブレーションは「素の（左右反転していない）」フレームで行う。
表示用の左右反転は呼び出し側が描画直前にのみ行うこと。虹彩比率は各目で画像座標の
min/max を使うため、左右どちらの目でも符号が常に揃い、鏡像セマンティクスを気にせずに済む。
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from enum import Enum

import numpy as np

# FaceLandmarker のモデル（478 ランドマーク＋虹彩）。初回のみダウンロードする。
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# --- ランドマークインデックス（MediaPipe FaceLandmarker, 虹彩込み 478 点）---

# 虹彩リング（中心 + 周囲 4 点）。平均を虹彩中心として使う。
LEFT_IRIS: tuple[int, ...] = (468, 469, 470, 471, 472)
RIGHT_IRIS: tuple[int, ...] = (473, 474, 475, 476, 477)

# 目の水平端（目頭・目尻）。左右の区別はせず min/max で使う。
LEFT_EYE_CORNERS: tuple[int, int] = (33, 133)
RIGHT_EYE_CORNERS: tuple[int, int] = (362, 263)

# 上下まぶた（縦方向）。
LEFT_EYE_LIDS: tuple[int, int] = (159, 145)
RIGHT_EYE_LIDS: tuple[int, int] = (386, 374)

# EAR（eye aspect ratio）用の縦ペア 2 組と横ペア。
LEFT_EAR_VERT: tuple[tuple[int, int], ...] = ((160, 144), (158, 153))
LEFT_EAR_HORIZ: tuple[int, int] = (33, 133)
RIGHT_EAR_VERT: tuple[tuple[int, int], ...] = ((385, 380), (387, 373))
RIGHT_EAR_HORIZ: tuple[int, int] = (362, 263)

# 頭部向きプロキシ用。
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10

_EPS = 1e-6

# 指標ベクトルの並び順（z スコア・キャリブレーションで共通）。
#   0: iris_h, 1: iris_v, 2: ear, 3: yaw, 4: pitch
_STD_FLOOR = np.array([0.01, 0.01, 0.02, 0.005, 0.005], dtype=np.float64)


@dataclass(frozen=True)
class GazeMetrics:
    """1 フレーム分の生の視線・頭部指標。"""

    iris_h: float  # 水平虹彩比（副信号／横ゲート）
    iris_v: float  # 垂直虹彩比（主信号: 下を見ると増える）
    ear: float  # まぶた開度（瞬き検出・共変量）
    yaw: float  # 頭部 yaw プロキシ（横ゲート）
    pitch: float  # 頭部 pitch プロキシ（主信号の片割れ）

    def as_vector(self) -> list[float]:
        return [self.iris_h, self.iris_v, self.ear, self.yaw, self.pitch]


def _eye_iris_ratios(
    pts: np.ndarray,
    iris_idx: tuple[int, ...],
    corner_idx: tuple[int, int],
    lid_idx: tuple[int, int],
) -> tuple[float, float]:
    """1 つの目について虹彩の水平・垂直比を返す（画像座標の min/max 基準）。"""
    iris = pts[list(iris_idx)].mean(axis=0)
    corner_x = pts[list(corner_idx), 0]
    lid_y = pts[list(lid_idx), 1]
    width = float(corner_x.max() - corner_x.min())
    height = float(lid_y.max() - lid_y.min())
    rh = float((iris[0] - corner_x.min()) / width) if width > _EPS else 0.5
    rv = float((iris[1] - lid_y.min()) / height) if height > _EPS else 0.5
    return rh, rv


def _eye_ear(
    pts: np.ndarray,
    vert_pairs: tuple[tuple[int, int], ...],
    horiz: tuple[int, int],
) -> float:
    """1 つの目の EAR（縦/横）。閉眼で 0 に近づく。"""
    width = abs(float(pts[horiz[0], 0] - pts[horiz[1], 0]))
    if width < _EPS:
        return 0.0
    vert = sum(abs(float(pts[a, 1] - pts[b, 1])) for a, b in vert_pairs) / len(vert_pairs)
    return vert / width


def _yaw_proxy(pts: np.ndarray) -> float:
    """鼻先の左右オフセットを顔幅で正規化した yaw プロキシ（距離不変）。"""
    nose_x = float(pts[NOSE_TIP, 0])
    left_x = float(pts[LEFT_CHEEK, 0])
    right_x = float(pts[RIGHT_CHEEK, 0])
    width = abs(right_x - left_x)
    if width < _EPS:
        return 0.0
    center = (left_x + right_x) / 2.0
    return (nose_x - center) / width


def _pitch_proxy(pts: np.ndarray) -> float:
    """鼻先の縦位置を目の高さ基準・顔の高さで正規化した pitch プロキシ。"""
    eye_y = (
        float(pts[LEFT_EYE_CORNERS[0], 1])
        + float(pts[LEFT_EYE_CORNERS[1], 1])
        + float(pts[RIGHT_EYE_CORNERS[0], 1])
        + float(pts[RIGHT_EYE_CORNERS[1], 1])
    ) / 4.0
    face_h = abs(float(pts[CHIN, 1]) - float(pts[FOREHEAD, 1]))
    if face_h < _EPS:
        return 0.0
    return (float(pts[NOSE_TIP, 1]) - eye_y) / face_h


def compute_metrics(pts: np.ndarray) -> GazeMetrics:
    """478×2 のピクセル座標ランドマークから 1 フレームの指標を計算する。"""
    lh, lv = _eye_iris_ratios(pts, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
    rh, rv = _eye_iris_ratios(pts, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
    left_ear = _eye_ear(pts, LEFT_EAR_VERT, LEFT_EAR_HORIZ)
    right_ear = _eye_ear(pts, RIGHT_EAR_VERT, RIGHT_EAR_HORIZ)
    return GazeMetrics(
        iris_h=(lh + rh) / 2.0,
        iris_v=(lv + rv) / 2.0,
        ear=(left_ear + right_ear) / 2.0,
        yaw=_yaw_proxy(pts),
        pitch=_pitch_proxy(pts),
    )


@dataclass(frozen=True)
class Calibration:
    """キャリブレーションで得た個人基準（各指標の平均と標準偏差）。"""

    mean: np.ndarray  # shape (5,)
    std: np.ndarray  # shape (5,)

    def zscores(self, m: GazeMetrics) -> np.ndarray:
        """指標を基準からの z スコア（標準偏差単位）に変換する。"""
        v = np.array(m.as_vector(), dtype=np.float64)
        return (v - self.mean) / np.maximum(self.std, _STD_FLOOR)


class Calibrator:
    """「カメラ（レンズ）を見ている」基準フレームを集めて Calibration を作る。"""

    def __init__(self, min_samples: int = 30, ear_floor: float = 0.12) -> None:
        self.min_samples = min_samples
        self.ear_floor = ear_floor
        self._samples: list[list[float]] = []

    def add(self, m: GazeMetrics) -> bool:
        """フレームを採用したら True。瞬き・閉眼フレームは除外する。"""
        if m.ear < self.ear_floor:
            return False
        self._samples.append(m.as_vector())
        return True

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        return self.count >= self.min_samples

    def finish(self) -> Calibration:
        if not self._samples:
            raise ValueError("キャリブレーションのサンプルがありません")
        arr = np.array(self._samples, dtype=np.float64)
        return Calibration(mean=arr.mean(axis=0), std=arr.std(axis=0))

    def reset(self) -> None:
        self._samples.clear()


class ContactState(Enum):
    CONTACT = "contact"
    NO_CONTACT = "no_contact"


@dataclass(frozen=True)
class ClassifierConfig:
    """分類器のしきい値。デバッグ表示を見ながら調整するための定数群。"""

    tau: float = 0.15  # EMA 時定数(秒)
    enter_z: float = 1.5  # CONTACT 進入の平滑偏差しきい値
    exit_z: float = 2.5  # CONTACT 退出の平滑偏差しきい値
    enter_ms: float = 150.0  # 進入に必要な継続時間
    exit_ms: float = 200.0  # 退出に必要な継続時間
    yaw_gate_z: float = 3.0  # |yaw z| がこれを超えたら強制 NO_CONTACT
    blink_ratio: float = 0.6  # EAR が基準のこの割合を下回れば瞬きとみなし判定を凍結
    w_iris_h: float = 0.3  # 偏差合成の重み（水平虹彩比）
    w_iris_v: float = 1.0  # 偏差合成の重み（垂直虹彩比＝主信号）
    w_pitch: float = 0.6  # 偏差合成の重み（頭部 pitch）


@dataclass(frozen=True)
class GazeDebug:
    """1 フレームの分類結果と、デバッグ表示用の中間値。"""

    metrics: GazeMetrics
    zscores: np.ndarray
    deviation: float
    smoothed: float
    state: ContactState
    yaw_gated: bool
    blink: bool = False


class GazeClassifier:
    """z スコア偏差を EMA 平滑化し、二重しきい値ヒステリシスで CONTACT を判定する。"""

    def __init__(self, calibration: Calibration, config: ClassifierConfig | None = None) -> None:
        self.calibration = calibration
        self.config = config or ClassifierConfig()
        self._smoothed: float | None = None
        self._state = ContactState.NO_CONTACT
        self._enter_elapsed = 0.0
        self._exit_elapsed = 0.0

    @property
    def state(self) -> ContactState:
        return self._state

    def _deviation(self, z: np.ndarray) -> float:
        c = self.config
        weighted = np.array(
            [c.w_iris_h * z[0], c.w_iris_v * z[1], c.w_pitch * z[4]],
            dtype=np.float64,
        )
        return float(np.sqrt(np.mean(weighted**2)))

    def update(self, m: GazeMetrics, dt: float) -> GazeDebug:
        z = self.calibration.zscores(m)

        # 瞬き中は判定を凍結（偽 NG・チラつき防止）。
        baseline_ear = float(self.calibration.mean[2])
        if m.ear < self.config.blink_ratio * baseline_ear:
            held = self._smoothed if self._smoothed is not None else 0.0
            return GazeDebug(
                metrics=m,
                zscores=z,
                deviation=held,
                smoothed=held,
                state=self._state,
                yaw_gated=False,
                blink=True,
            )

        dev = self._deviation(z)
        alpha = float(1.0 - np.exp(-dt / self.config.tau)) if dt > 0 else 1.0
        if self._smoothed is None:
            self._smoothed = dev
        else:
            self._smoothed += alpha * (dev - self._smoothed)

        yaw_gated = abs(float(z[3])) > self.config.yaw_gate_z
        self._apply_hysteresis(dt, yaw_gated)
        return GazeDebug(
            metrics=m,
            zscores=z,
            deviation=dev,
            smoothed=self._smoothed,
            state=self._state,
            yaw_gated=yaw_gated,
        )

    def _apply_hysteresis(self, dt: float, yaw_gated: bool) -> None:
        c = self.config
        smoothed = self._smoothed if self._smoothed is not None else 0.0
        if yaw_gated:
            self._state = ContactState.NO_CONTACT
            self._enter_elapsed = 0.0
            self._exit_elapsed = 0.0
            return
        if self._state is ContactState.NO_CONTACT:
            if smoothed < c.enter_z:
                self._enter_elapsed += dt
                if self._enter_elapsed * 1000.0 >= c.enter_ms:
                    self._state = ContactState.CONTACT
                    self._exit_elapsed = 0.0
            else:
                self._enter_elapsed = 0.0
        elif smoothed > c.exit_z:
            self._exit_elapsed += dt
            if self._exit_elapsed * 1000.0 >= c.exit_ms:
                self._state = ContactState.NO_CONTACT
                self._enter_elapsed = 0.0
        else:
            self._exit_elapsed = 0.0

    def reset_state(self) -> None:
        self._smoothed = None
        self._state = ContactState.NO_CONTACT
        self._enter_elapsed = 0.0
        self._exit_elapsed = 0.0


def drift_direction(z: np.ndarray, threshold: float = 1.0) -> str:
    """NO_CONTACT 時に視線が外れた向きを返す（"down"/"up"/"left"/"right"/"center"）。

    面接で最も価値が高いのは "down"（画面・手元を見ている）の検出。
    """
    vert = float(z[1])  # iris_v: 正で下向き
    horiz = float(z[0])  # iris_h: 画像座標での横ずれ
    if abs(vert) < threshold and abs(horiz) < threshold:
        return "center"
    if abs(vert) >= abs(horiz):
        return "down" if vert > 0 else "up"
    return "right" if horiz > 0 else "left"


def ensure_model(path: str = _MODEL_PATH) -> str:
    """FaceLandmarker のモデルを用意する（無ければダウンロード）。パスを返す。"""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"モデルをダウンロード中: {_MODEL_URL}")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print(f"保存しました: {path}")
    return path


class FaceMeshTracker:
    """MediaPipe Tasks の FaceLandmarker（VIDEO モード）の薄いラッパ。

    素の RGB フレームを受け、478×2 のピクセル座標ランドマーク配列を返す
    （顔が無ければ None）。mediapipe は遅延 import するため、純粋ロジックの
    単体テストでは本クラスをインスタンス化しない限り mediapipe を要求しない。
    """

    def __init__(
        self,
        model_path: str | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp  # 遅延 import
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=ensure_model(model_path or _MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        # VIDEO モードはタイムスタンプが厳密に増加する必要があるためクランプする。
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        height, width = rgb.shape[:2]
        pts = np.empty((len(landmarks), 2), dtype=np.float64)
        for i, p in enumerate(landmarks):
            pts[i, 0] = p.x * width
            pts[i, 1] = p.y * height
        return pts

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceMeshTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
