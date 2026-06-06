import { expect, test } from "bun:test";
import { headPoseFromMatrix } from "../src/lib/headpose";

// 行優先 3×3 回転（＋並進）を MediaPipe の列優先 16 要素に変換。
function colMajor(R: number[][], t: number[] = [0, 0, 0]): number[] {
  const m = new Array<number>(16).fill(0);
  m[15] = 1;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      m[col * 4 + row] = R[row][col];
    }
  }
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

test("pure yaw rotation", () => {
  const th = 0.3;
  const ry = [
    [Math.cos(th), 0, Math.sin(th)],
    [0, 1, 0],
    [-Math.sin(th), 0, Math.cos(th)],
  ];
  const hp = headPoseFromMatrix(colMajor(ry));
  expect(hp.yaw).toBeCloseTo(0.3, 5);
  expect(hp.pitch).toBeCloseTo(0, 5);
  expect(hp.roll).toBeCloseTo(0, 5);
});

test("pure pitch rotation", () => {
  const ph = -0.25;
  const rx = [
    [1, 0, 0],
    [0, Math.cos(ph), -Math.sin(ph)],
    [0, Math.sin(ph), Math.cos(ph)],
  ];
  const hp = headPoseFromMatrix(colMajor(rx));
  expect(hp.pitch).toBeCloseTo(-0.25, 5);
  expect(hp.yaw).toBeCloseTo(0, 5);
});

test("translation is read from the last column", () => {
  const identity = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const hp = headPoseFromMatrix(colMajor(identity, [1, 2, 3]));
  expect(hp.tx).toBe(1);
  expect(hp.ty).toBe(2);
  expect(hp.tz).toBe(3);
});
