// 特徴ベクトル生成（純粋）。頭部姿勢で de-rotate した「eye-in-head（眼球内）視線」を主役にし、
// 画像座標の虹彩比率は使わない。これにより頭の動きと目の動きを分離する。

import { applyTranspose, headPoseFromMatrix, rotationMatrix } from "./headpose";

export type Point = { x: number; y: number };
export type Point3 = { x: number; y: number; z: number };

export interface FaceFrame {
  landmarks: Point3[]; // x,y はピクセル換算、z は p.z*W（x とほぼ同スケール）
  blend: Record<string, number>;
  matrix: number[]; // 4×4 顔変換行列（列優先 16）
}

const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
// 目輪郭（中心推定用）: 目頭・目尻＋上下まぶた数点
const LEFT_EYE = [33, 133, 159, 145, 160, 144, 158, 153];
const RIGHT_EYE = [362, 263, 386, 374, 385, 380, 387, 373];
const LEFT_CORNERS: [number, number] = [33, 133];
const RIGHT_CORNERS: [number, number] = [362, 263];

const EYE_LOOK_KEYS = [
  "eyeLookInLeft",
  "eyeLookOutLeft",
  "eyeLookUpLeft",
  "eyeLookDownLeft",
  "eyeLookInRight",
  "eyeLookOutRight",
  "eyeLookUpRight",
  "eyeLookDownRight",
];

export interface FrameAnalysis {
  features: number[];
  gaze: { x: number; y: number }; // 頭部不変の eye-in-head 視線（固視検出・デバッグ用）
  blink: number;
  pose: { yaw: number; pitch: number; roll: number };
}

function mean3(pts: Point3[], idx: number[]): Point3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (const i of idx) {
    x += pts[i].x;
    y += pts[i].y;
    z += pts[i].z;
  }
  return { x: x / idx.length, y: y / idx.length, z: z / idx.length };
}

// 画像座標(y 下向き)と顔行列フレーム(y 上向き・右手系)の規約差で de-rotate が崩れる場合に
// 調整するための符号。既定は現状維持。実機で「カメラを固視して頭を回すと eye(gx,gy) が頭に
// 追従してしまう」場合、まず y を -1 にする（README の受け入れテスト参照）。
const GAZE_SIGN = { x: 1, y: 1, z: 1 };

function eyeInHead(
  pts: Point3[],
  iris: number[],
  eye: number[],
  corners: [number, number],
  R: number[][],
): [number, number] {
  const c = mean3(pts, iris);
  const e = mean3(pts, eye);
  // 頭部回転を除去（Rᵀ）。規約差は GAZE_SIGN で吸収可能。
  const gaze = applyTranspose(R, [
    GAZE_SIGN.x * (c.x - e.x),
    GAZE_SIGN.y * (c.y - e.y),
    GAZE_SIGN.z * (c.z - e.z),
  ]);
  const eyeWidth =
    Math.hypot(pts[corners[0]].x - pts[corners[1]].x, pts[corners[0]].y - pts[corners[1]].y) || 1;
  return [gaze[0] / eyeWidth, gaze[1] / eyeWidth];
}

export function analyzeFrame(frame: FaceFrame): FrameAnalysis {
  const pose = headPoseFromMatrix(frame.matrix);
  const R = rotationMatrix(frame.matrix);
  const [lx, ly] = eyeInHead(frame.landmarks, LEFT_IRIS, LEFT_EYE, LEFT_CORNERS, R);
  const [rx, ry] = eyeInHead(frame.landmarks, RIGHT_IRIS, RIGHT_EYE, RIGHT_CORNERS, R);
  const gx = (lx + rx) / 2;
  const gy = (ly + ry) / 2;
  const eye = EYE_LOOK_KEYS.map((k) => frame.blend[k] ?? 0);
  const features = [gx, gy, ...eye, pose.yaw, pose.pitch, pose.roll, pose.tx, pose.ty, pose.tz];
  const blink = Math.max(frame.blend.eyeBlinkLeft ?? 0, frame.blend.eyeBlinkRight ?? 0);
  return { features, gaze: { x: gx, y: gy }, blink, pose };
}

// 特徴次元: gazeX,gazeY + eyeLook×8 + yaw,pitch,roll + tx,ty,tz
export const FEATURE_DIM = 2 + EYE_LOOK_KEYS.length + 3 + 3;
