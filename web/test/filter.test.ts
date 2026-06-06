import { expect, test } from "bun:test";
import { OneEuroFilter } from "../src/lib/filter";

test("first value passes through", () => {
  const f = new OneEuroFilter();
  expect(f.filter(5, 0)).toBe(5);
});

test("constant input stays constant", () => {
  const f = new OneEuroFilter(1, 0.1);
  f.filter(2, 0);
  for (let i = 1; i < 10; i++) {
    expect(f.filter(2, i * 0.05)).toBeCloseTo(2, 6);
  }
});

test("reduces amplitude of noisy input", () => {
  const f = new OneEuroFilter(0.5, 0.0);
  const inputs = [0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
  const outs = inputs.map((x, i) => f.filter(x, i * 0.05));
  const inAmp = Math.max(...inputs) - Math.min(...inputs);
  const outAmp = Math.max(...outs) - Math.min(...outs);
  expect(outAmp).toBeLessThan(inAmp);
});
