// 頭部姿勢（純粋）。MediaPipe の顔変換行列（4×4・列優先 16 要素）から
// yaw/pitch/roll と並進（頭位置）を取り出す。
// 符号・規約は実機のデバッグ表示で確認する前提（回帰には滑らかな単調性があれば足りる）。

export interface HeadPose {
  yaw: number; // Y 軸回り（左右）
  pitch: number; // X 軸回り（上下）
  roll: number; // Z 軸回り（傾き）
  tx: number;
  ty: number;
  tz: number; // 並進（頭位置・距離）
}

export function headPoseFromMatrix(m: number[]): HeadPose {
  // 列優先: R[row][col] = m[col*4 + row]
  const r = (row: number, col: number): number => m[col * 4 + row];
  const r02 = r(0, 2);
  const r10 = r(1, 0);
  const r11 = r(1, 1);
  const r12 = r(1, 2);
  const r22 = r(2, 2);

  const yaw = Math.atan2(r02, r22);
  const pitch = Math.atan2(-r12, Math.hypot(r02, r22));
  const roll = Math.atan2(r10, r11);
  return { yaw, pitch, roll, tx: m[12], ty: m[13], tz: m[14] };
}

export const IDENTITY16: number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

// 列優先 4×4 から 3×3 回転行列 R を取り出す（R[row][col]）。
export function rotationMatrix(m: number[]): number[][] {
  const r = (row: number, col: number): number => m[col * 4 + row];
  return [
    [r(0, 0), r(0, 1), r(0, 2)],
    [r(1, 0), r(1, 1), r(1, 2)],
    [r(2, 0), r(2, 1), r(2, 2)],
  ];
}

// Rᵀ·v（頭部回転を除去して頭部固定フレームへ）。R は回転なので Rᵀ=R⁻¹。
export function applyTranspose(
  R: number[][],
  v: [number, number, number],
): [number, number, number] {
  return [
    R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
    R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
  ];
}

// R·v（順方向回転。テスト用）。
export function applyRotation(
  R: number[][],
  v: [number, number, number],
): [number, number, number] {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}
