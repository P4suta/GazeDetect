import { expect, test } from "bun:test";
import { ContactState } from "../src/lib/gaze";
import { SessionStats } from "../src/lib/stats";

test("face lost pauses visible timer", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  s.update(1, null);
  expect(s.totalTime).toBe(2);
  expect(s.visibleTime).toBe(1);
  expect(s.contactTime).toBe(1);
  expect(s.currentStreak).toBe(0);
});

test("contact ratio", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  s.update(1, ContactState.Contact);
  s.update(2, ContactState.NoContact, "down");
  expect(s.contactRatio).toBe(0.5);
});

test("longest streak tracks max", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  s.update(1, ContactState.Contact);
  s.update(1, ContactState.NoContact, "down");
  s.update(1, ContactState.Contact);
  expect(s.longestStreak).toBe(2);
  expect(s.currentStreak).toBe(1);
});

test("dominant drift", () => {
  const s = new SessionStats();
  s.update(3, ContactState.NoContact, "down");
  s.update(1, ContactState.NoContact, "left");
  expect(s.dominantDrift()).toBe("down");
});

test("dominant drift none when no drift", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  expect(s.dominantDrift()).toBeNull();
});

test("reset clears everything", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  s.update(1, ContactState.NoContact, "down");
  s.reset();
  expect(s.totalTime).toBe(0);
  expect(s.visibleTime).toBe(0);
  expect(s.contactTime).toBe(0);
  expect(s.longestStreak).toBe(0);
  expect(Object.values(s.driftTime).every((v) => v === 0)).toBe(true);
});

test("summary lines include ratio", () => {
  const s = new SessionStats();
  s.update(1, ContactState.Contact);
  const lines = s.summaryLines();
  expect(lines.some((l) => l.includes("維持率"))).toBe(true);
});
