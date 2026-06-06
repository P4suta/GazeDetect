import { expect, test } from "bun:test";
import { ContactState, driftDirection, EyeContactClassifier, GazeModel } from "../src/lib/gaze";
import { Standardizer } from "../src/lib/linalg";

// 特徴 [a,b] をそのまま PoR(a,b) に写す自明モデル（scaler 恒等、weights= bias0 + 各次元）。
function trivialModel(cam = { x: 0.5, y: 0.5 }): GazeModel {
  const scaler = new Standardizer([0, 0], [1, 1]);
  const W = [
    [0, 0],
    [1, 0],
    [0, 1],
  ];
  return new GazeModel(scaler, W, cam);
}

test("gaze model predicts PoR", () => {
  const p = trivialModel().predict([0.3, 0.7]);
  expect(p.x).toBeCloseTo(0.3, 6);
  expect(p.y).toBeCloseTo(0.7, 6);
});

test("eye contact enters near camera, exits when far", () => {
  const clf = new EyeContactClassifier(trivialModel({ x: 0.5, y: 0.5 }));
  for (let i = 0; i < 10; i++) {
    clf.update([0.5, 0.5], 0, 0.05);
  }
  expect(clf.currentState).toBe(ContactState.Contact);

  for (let i = 0; i < 12; i++) {
    clf.update([0.95, 0.5], 0, 0.05);
  }
  expect(clf.currentState).toBe(ContactState.NoContact);
});

test("blink freezes the decision", () => {
  const clf = new EyeContactClassifier(trivialModel());
  for (let i = 0; i < 10; i++) {
    clf.update([0.5, 0.5], 0, 0.05);
  }
  expect(clf.currentState).toBe(ContactState.Contact);

  const r = clf.update([0.95, 0.5], 0.9, 0.05);
  expect(r.blink).toBe(true);
  expect(clf.currentState).toBe(ContactState.Contact);
});

test("drift direction relative to camera point", () => {
  const cam = { x: 0.5, y: 0.5 };
  expect(driftDirection({ x: 0.5, y: 0.9 }, cam)).toBe("down");
  expect(driftDirection({ x: 0.5, y: 0.1 }, cam)).toBe("up");
  expect(driftDirection({ x: 0.9, y: 0.5 }, cam)).toBe("right");
  expect(driftDirection({ x: 0.1, y: 0.5 }, cam)).toBe("left");
  expect(driftDirection({ x: 0.52, y: 0.52 }, cam)).toBe("center");
});
