import { expect, test } from "bun:test";
import { CalibrationEngine } from "../src/lib/calibengine";
import { AdaptiveThreshold } from "../src/lib/fixation";

test("adaptive threshold is lenient until enough samples, then ~percentile", () => {
  const a = new AdaptiveThreshold(6, 0.5, 5);
  expect(a.value()).toBe(Number.POSITIVE_INFINITY);
  for (let i = 0; i < 10; i++) {
    a.push(i * 0.01, i * 0.1); // 0 .. 0.09
  }
  const v = a.value();
  expect(v).toBeGreaterThan(0.02);
  expect(v).toBeLessThan(0.08);
});

// 一定の安定注視（dispersion≈0）を与えると、連続良好固視でロックされる。
test("a sustained steady fixation locks a point", () => {
  const eng = new CalibrationEngine();
  let t = 0;
  const dt = 0.033;
  let target = { x: 0.5, y: 0.5 };
  let locked = false;
  for (let i = 0; i < 300 && !locked; i++) {
    t += dt;
    const s = eng.feed(
      { features: [target.x, target.y], gaze: { x: 0.1, y: 0.2 }, blink: 0 },
      dt,
      t,
    );
    if (s.target) {
      target = { x: s.target.x, y: s.target.y };
    }
    if (s.justLocked) {
      locked = true;
    }
  }
  expect(locked).toBe(true);
});

// 良好固視で hold を貯めた後、瞬き（無効フレーム）を続けると hold が減衰する。
test("breaking fixation drains the hold progress", () => {
  const eng = new CalibrationEngine();
  const dt = 0.033;
  const target = { x: 0.5, y: 0.5 };
  const good = { features: [target.x, target.y], gaze: { x: 0.1, y: 0.2 }, blink: 0 };
  const blinking = { features: [target.x, target.y], gaze: { x: 0.1, y: 0.2 }, blink: 1 };
  let t = 0;
  let s = eng.feed(good, dt, t);
  for (let i = 0; i < 28; i++) {
    t += dt;
    s = eng.feed(good, dt, t);
  }
  const before = s.holdProgress;
  expect(before).toBeGreaterThan(0.1);
  for (let i = 0; i < 12; i++) {
    t += dt;
    s = eng.feed(blinking, dt, t);
  }
  expect(s.holdProgress).toBeLessThan(before);
});

// 通し: 全点ロック → fit → held-out 検証まで完了し、誤差が小さい。
test("a full steady run completes with a model and small held-out error", () => {
  const eng = new CalibrationEngine();
  let t = 0;
  const dt = 0.033;
  let target = { x: 0.5, y: 0.5 };
  let done = false;
  let accuracy = Number.POSITIVE_INFINITY;
  for (let i = 0; i < 8000 && !done; i++) {
    t += dt;
    const s = eng.feed(
      { features: [target.x, target.y], gaze: { x: 0.1, y: 0.2 }, blink: 0 },
      dt,
      t,
    );
    if (s.target) {
      target = { x: s.target.x, y: s.target.y };
    }
    if (s.done) {
      done = true;
      expect(s.model).not.toBeNull();
      accuracy = s.accuracy ?? Number.POSITIVE_INFINITY;
    }
  }
  expect(done).toBe(true);
  expect(accuracy).toBeLessThan(0.1);
});
