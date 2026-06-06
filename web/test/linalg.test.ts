import { expect, test } from "bun:test";
import { applyWeights, ridgeFit, Standardizer, solveLinearSystem } from "../src/lib/linalg";

test("solveLinearSystem solves a 2x2 system", () => {
  // 2a+b=1, a+3b=2 -> a=0.2, b=0.6
  const x = solveLinearSystem(
    [
      [2, 1],
      [1, 3],
    ],
    [[1], [2]],
  );
  expect(x[0][0]).toBeCloseTo(0.2, 6);
  expect(x[1][0]).toBeCloseTo(0.6, 6);
});

test("ridgeFit recovers a known linear mapping", () => {
  const wTrue = [
    [0.5, -0.2],
    [1.0, 0.3],
    [-0.4, 0.8],
  ];
  const rows = [
    [1, 0.1, 0.2],
    [1, 0.4, -0.1],
    [1, -0.3, 0.5],
    [1, 0.2, 0.2],
    [1, -0.1, -0.4],
    [1, 0.5, 0.1],
  ];
  const Y = rows.map((x) => applyWeights(wTrue, x));
  const W = ridgeFit(rows, Y, 1e-6);
  for (const x of rows) {
    const p = applyWeights(W, x);
    const t = applyWeights(wTrue, x);
    expect(p[0]).toBeCloseTo(t[0], 3);
    expect(p[1]).toBeCloseTo(t[1], 3);
  }
});

test("standardizer zero-means each dimension", () => {
  const s = Standardizer.fit([
    [0, 10],
    [2, 20],
    [4, 30],
  ]);
  expect(s.mean[0]).toBeCloseTo(2, 6);
  const t = s.transform([2, 20]);
  expect(t[0]).toBeCloseTo(0, 6);
  expect(t[1]).toBeCloseTo(0, 6);
});

test("standardizer handles zero-variance dimension", () => {
  const s = Standardizer.fit([
    [5, 1],
    [5, 2],
    [5, 3],
  ]);
  expect(s.transform([5, 2])[0]).toBe(0); // std=1 fallback -> (5-5)/1 = 0
});
