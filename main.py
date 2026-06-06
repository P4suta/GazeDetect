"""GazeDetect エントリポイント。カメラループ・状態機械・CLI。

使い方は README を参照。操作キー: q 終了 / c 再キャリブ / d デバッグ / r リセット。
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from gaze import (
    Calibrator,
    ContactState,
    FaceMeshTracker,
    GazeClassifier,
    compute_metrics,
    drift_direction,
)
from overlay import OverlayRenderer, OverlayState
from stats import SessionStats

WINDOW = "GazeDetect"
_MAX_DT = 0.1  # 1 フレームの dt 上限（処理停止後の過剰加算を防ぐ）
_WARMUP_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラ目線検知（面接練習用）")
    parser.add_argument(
        "--source",
        default="0",
        help="カメラ番号（例: 0, 1）または動画ファイルパス",
    )
    parser.add_argument("--width", type=int, default=None, help="キャプチャ幅")
    parser.add_argument("--height", type=int, default=None, help="キャプチャ高さ")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="表示の左右反転を無効化（既定は鏡像表示）",
    )
    parser.add_argument(
        "--calib-seconds",
        type=float,
        default=3.0,
        help="キャリブレーション時間（秒）",
    )
    return parser.parse_args()


def open_capture(source: int | str, width: int | None, height: int | None) -> cv2.VideoCapture:
    """カメラ/動画を開く。Windows のカメラは DirectShow を優先しフォールバックする。"""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def mirror_points(pts: np.ndarray, width: int) -> np.ndarray:
    """ランドマークを表示用（左右反転）座標系へ変換する。"""
    mirrored = pts.copy()
    mirrored[:, 0] = (width - 1) - mirrored[:, 0]
    return mirrored


def window_closed() -> bool:
    """ウィンドウの×ボタンで閉じられたか。"""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run(args: argparse.Namespace) -> None:
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    mirror = not args.no_mirror

    cap = open_capture(source, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"カメラ/動画を開けませんでした: {source!r}")

    for _ in range(_WARMUP_FRAMES):  # 露出が安定するまで数フレーム捨てる
        cap.read()

    tracker = FaceMeshTracker()
    renderer = OverlayRenderer()
    stats = SessionStats()

    mode = "WAIT_FOR_FACE"
    calibrator = Calibrator()
    calib_elapsed = 0.0
    classifier: GazeClassifier | None = None
    show_debug = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("起動しました。カメラのレンズを見てキャリブレーションしてください。")
    print("操作: q 終了 / c 再キャリブ / d デバッグ / r リセット")

    fps = 0.0
    start = time.perf_counter()
    prev = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = min(now - prev, _MAX_DT)
            prev = now
            timestamp_ms = int((now - start) * 1000)
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            width = frame.shape[1]

            # --- 推論は「素の」フレームで行う ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pts = tracker.process(rgb, timestamp_ms)
            face_visible = pts is not None
            metrics = compute_metrics(pts) if pts is not None else None

            # --- 状態機械 ---
            debug = None
            if mode == "WAIT_FOR_FACE":
                if face_visible:
                    mode = "CALIBRATING"
                    calibrator.reset()
                    calib_elapsed = 0.0
            elif mode == "CALIBRATING":
                if metrics is not None:
                    calibrator.add(metrics)
                    calib_elapsed += dt
                    # 通常は時間経過＋必要枚数。EAR が低くサンプルが集まりにくい環境でも
                    # 永久にハングしないよう、2 倍の時間で最低枚数あれば打ち切る。
                    ready = calib_elapsed >= args.calib_seconds and calibrator.ready
                    timed_out = calib_elapsed >= 2 * args.calib_seconds and calibrator.count >= 10
                    if ready or timed_out:
                        classifier = GazeClassifier(calibrator.finish())
                        mode = "ACTIVE"
                        print(
                            f"キャリブレーション完了（{calibrator.count} フレーム）。"
                            "計測を開始します。"
                        )
            elif mode == "ACTIVE" and classifier is not None:
                if metrics is not None:
                    debug = classifier.update(metrics, dt)
                    drift = (
                        drift_direction(debug.zscores)
                        if debug.state is ContactState.NO_CONTACT
                        else None
                    )
                    stats.update(dt, debug.state, drift)
                else:
                    stats.update(dt, None)

            # --- 表示 ---
            display = cv2.flip(frame, 1) if mirror else frame.copy()
            pts_display = None
            if pts is not None:
                pts_display = mirror_points(pts, width) if mirror else pts

            state = OverlayState(
                mode=mode,
                debug=debug,
                face_visible=face_visible,
                calib_progress=min(calib_elapsed / args.calib_seconds, 1.0),
                calib_remaining=max(args.calib_seconds - calib_elapsed, 0.0),
                calib_count=calibrator.count,
                calib_min=calibrator.min_samples,
                show_debug=show_debug,
                fps=fps,
                landmarks_mirrored=pts_display,
            )
            display = renderer.render(display, state, stats)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / Esc
                break
            if key == ord("d"):
                show_debug = not show_debug
            elif key == ord("r"):
                stats.reset()
            elif key == ord("c"):
                mode = "CALIBRATING"
                calibrator.reset()
                calib_elapsed = 0.0
                classifier = None
                print("再キャリブレーションします。カメラのレンズを見てください。")

            if window_closed():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    for line in stats.summary_lines():
        print(line)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
