import { expect, test } from "bun:test";
import { applyRotation, applyTranspose, rotationMatrix } from "../src/lib/headpose";
import { applyWeights, ridgeFitWeighted } from "../src/lib/linalg";

function colMajor(R: number[][]): number[] {
  const m = new Array<number>(16).fill(0);
  m[15] = 1;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      m[col * 4 + row] = R[row][col];
    }
  }
  return m;
}

function rotY(th: number): number[][] {
  return [
    [Math.cos(th), 0, Math.sin(th)],
    [0, 1, 0],
    [-Math.sin(th), 0, Math.cos(th)],
  ];
}

test("rotationMatrix extracts the 3x3 rotation", () => {
  const R = rotationMatrix(colMajor(rotY(0.4)));
  expect(R[0][2]).toBeCloseTo(Math.sin(0.4), 6);
  expect(R[2][0]).toBeCloseTo(-Math.sin(0.4), 6);
});

test("de-rotation recovers the head-frame vector (head invariance)", () => {
  const R = rotationMatrix(colMajor(rotY(0.5)));
  const v: [number, number, number] = [0.2, -0.1, 0.05];
  const rotated = applyRotation(R, v); // 頭が回った状態
  const back = applyTranspose(R, rotated); // Rᵀ で de-rotate
  expect(back[0]).toBeCloseTo(v[0], 6);
  expect(back[1]).toBeCloseTo(v[1], 6);
  expect(back[2]).toBeCloseTo(v[2], 6);
});

test("weighted ridge favors high-weight samples", () => {
  // 同じ特徴 [bias=1, 0] に矛盾する目標 0 と 1。重み 10 の側（0）に寄るはず。
  const X = [
    [1, 0],
    [1, 0],
  ];
  const Y = [[0], [1]];
  const W = ridgeFitWeighted(X, Y, [10, 1], 1e-6);
  const p = applyWeights(W, [1, 0]);
  expect(p[0]).toBeLessThan(0.2);
});
