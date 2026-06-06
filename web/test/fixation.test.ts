import { expect, test } from "bun:test";
import { FixationDetector } from "../src/lib/fixation";

test("stable input is detected as a fixation", () => {
  const f = new FixationDetector(0.2, 0.06);
  let stable = false;
  for (let i = 0; i < 10; i++) {
    stable = f.push(0.1 + (i % 2) * 0.005, 0.2, i * 0.03);
  }
  expect(stable).toBe(true);
});

test("jumpy input is not a fixation", () => {
  const f = new FixationDetector(0.2, 0.06);
  let stable = true;
  for (let i = 0; i < 10; i++) {
    stable = f.push((i % 2) * 0.5, (i % 3) * 0.3, i * 0.03);
  }
  expect(stable).toBe(false);
});

test("quality is high for a tight fixation", () => {
  const f = new FixationDetector(0.2, 0.06);
  for (let i = 0; i < 10; i++) {
    f.push(0.1, 0.2, i * 0.03);
  }
  expect(f.quality()).toBeGreaterThan(0.8);
});
