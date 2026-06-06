import { expect, test } from "bun:test";
import {
  CALIB_TARGETS,
  CAMERA_POINT,
  CAMERA_TARGET_ID,
  CalibrationCollector,
  SCREEN_TARGETS,
} from "../src/lib/calibration";
import { DEFAULT_EC_CONFIG } from "../src/lib/gaze";

// 特徴 = [targetX, targetY]（恒等写像）。各点に微小ジッタの正常サンプル＋粗い外れ値を混ぜ、
// fit 経路（外れ値除去 → 標準化 → リッジ → カメラ点抽出）を丸ごと検証する。
function buildCollector(): CalibrationCollector {
  const c = new CalibrationCollector();
  for (const t of CALIB_TARGETS) {
    for (let i = 0; i < 20; i++) {
      const j = ((i % 5) - 2) * 0.002;
      c.add(t.id, [t.x + j, t.y - j]);
    }
    // 粗い外れ値（MAD 除去で落ちるはず）
    c.add(t.id, [t.x + 0.5, t.y + 0.5]);
    c.add(t.id, [t.x - 0.6, t.y - 0.4]);
  }
  return c;
}

test("fit recovers the target mapping despite outliers", () => {
  const model = buildCollector().fit(CALIB_TARGETS, 0.01);
  for (const t of CALIB_TARGETS) {
    const p = model.predict([t.x, t.y]);
    expect(Math.hypot(p.x - t.x, p.y - t.y)).toBeLessThan(0.08);
  }
});

test("camera point is extracted and camera-gaze maps near it", () => {
  const model = buildCollector().fit(CALIB_TARGETS, 0.01);
  expect(model.camPoint.x).toBeCloseTo(CAMERA_POINT.x, 6);
  expect(model.camPoint.y).toBeCloseTo(CAMERA_POINT.y, 6);

  const p = model.predict([CAMERA_POINT.x, CAMERA_POINT.y]);
  const distance = Math.hypot(p.x - model.camPoint.x, p.y - model.camPoint.y);
  expect(distance).toBeLessThan(DEFAULT_EC_CONFIG.enterR);
});

test("screen-center gaze is far from the camera point (discriminable)", () => {
  const model = buildCollector().fit(CALIB_TARGETS, 0.01);
  const center = model.predict([0.5, 0.5]);
  const distance = Math.hypot(center.x - model.camPoint.x, center.y - model.camPoint.y);
  // カメラ(y=-0.12) と画面中央(y=0.5) は退出しきい値を超えて離れている＝判別可能
  expect(distance).toBeGreaterThan(DEFAULT_EC_CONFIG.exitR);
});

// camPoint は固定値ではなく「レンズ注視の視線」を学習済みモデルで外挿して実測する。
function modelWithCamGaze(camY: number): ReturnType<CalibrationCollector["fit"]> {
  const c = new CalibrationCollector();
  for (const t of SCREEN_TARGETS) {
    for (let i = 0; i < 20; i++) {
      const j = ((i % 5) - 2) * 0.002;
      c.add(t.id, [t.x + j, t.y - j]); // グリッドは恒等写像
    }
  }
  for (let i = 0; i < 20; i++) {
    c.add(CAMERA_TARGET_ID, [0.5, camY]); // レンズ注視の視線特徴
  }
  return c.fit(SCREEN_TARGETS, 0.01, CAMERA_TARGET_ID);
}

test("camera point is measured from lens-gaze, not hardcoded", () => {
  const a = modelWithCamGaze(-0.2);
  const b = modelWithCamGaze(-0.5);
  // より上を見た方が camPoint も上に出る（入力に追従＝実測）
  expect(b.camPoint.y).toBeLessThan(a.camPoint.y);
  // 画面の外（上）に出る、かつハードコード -0.12 固定ではない
  expect(a.camPoint.y).toBeLessThan(0);
  expect(Math.abs(a.camPoint.y - CAMERA_POINT.y)).toBeGreaterThan(0.01);
});
