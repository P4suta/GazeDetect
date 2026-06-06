<script lang="ts">
import { onDestroy, onMount } from "svelte";
import {
  CalibrationEngine,
  type DisplayTarget,
  type Phase,
} from "./lib/calibengine";
import { analyzeFrame } from "./lib/features";
import {
  ContactState,
  driftDirection,
  EyeContactClassifier,
  type EyeContactDebug,
  type GazeModel,
} from "./lib/gaze";
import { drawFrame } from "./lib/overlay";
import { Beeper } from "./lib/sound";
import { SessionStats } from "./lib/stats";
import { FaceTracker } from "./lib/tracker";
import Controls from "./components/Controls.svelte";
import Hud from "./components/Hud.svelte";

type Mode = "WAIT_FOR_FACE" | "CALIBRATING" | "ACTIVE";

let videoEl: HTMLVideoElement;
let canvasEl: HTMLCanvasElement;

let mode = $state<Mode>("WAIT_FOR_FACE");
let faceVisible = $state(false);
let showDebug = $state(false);
let ec = $state<EyeContactDebug | null>(null);
let fps = $state(0);
let errorMsg = $state<string | null>(null);
let statsView = $state({ ratio: 0, longest: 0, current: 0 });
let accuracy = $state<number | null>(null);
let gazeDot = $state<{ x: number; y: number } | null>(null);
let frame = $state<{ gx: number; gy: number; yaw: number; pitch: number } | null>(null);

// キャリブ表示用
let calTarget = $state<DisplayTarget | null>(null);
let calHold = $state(0);
let calPhase = $state<Phase>("settle");
let calDisp = $state(0);
let calThresh = $state(0);
let calGood = $state(false);
let calIndex = $state(0);
let calTotal = $state(0);

// 非リアクティブな実体
let tracker: FaceTracker | null = null;
let engine: CalibrationEngine | null = null;
const beeper = new Beeper();
let model: GazeModel | null = null;
let classifier: EyeContactClassifier | null = null;
const stats = new SessionStats();
let prevPhase: Phase = "settle";
let rafId = 0;
let prev = 0;
let startTs = 0;
let running = false;

const contactClass = $derived.by(() => {
  if (mode === "CALIBRATING") {
    return "calib";
  }
  if (mode === "ACTIVE" && faceVisible && ec) {
    return ec.state === ContactState.Contact ? "ok" : "ng";
  }
  return "idle";
});

const calKind = $derived(calTarget?.kind ?? "calibrate");
const calPos = $derived(calTarget ? displayXY(calTarget) : { x: 50, y: 50 });
const calIsCamera = $derived(calTarget?.isCamera ?? false);

function clamp(v: number, a: number, b: number): number {
  return Math.min(b, Math.max(a, v));
}

function displayXY(t: DisplayTarget): { x: number; y: number } {
  return { x: clamp(t.x, 0.06, 0.94) * 100, y: clamp(t.y, 0.08, 0.92) * 100 };
}

onMount(async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    beeper.resume();
    tracker = await FaceTracker.create();
    startTs = performance.now();
    prev = startTs;
    running = true;
    rafId = requestAnimationFrame(loop);
  } catch (e) {
    errorMsg = e instanceof Error ? e.message : String(e);
  }
});

onDestroy(() => {
  running = false;
  cancelAnimationFrame(rafId);
  tracker?.close();
  const stream = videoEl?.srcObject as MediaStream | null;
  for (const track of stream?.getTracks() ?? []) {
    track.stop();
  }
});

function startCalibration(): void {
  engine = new CalibrationEngine();
  prevPhase = "settle";
  classifier = null;
  model = null;
  errorMsg = null;
  mode = "CALIBRATING";
}

function stepCalibration(
  analysis: ReturnType<typeof analyzeFrame> | null,
  nowSec: number,
  dt: number,
): void {
  if (!engine) {
    return;
  }
  const input = analysis
    ? { features: analysis.features, gaze: analysis.gaze, blink: analysis.blink }
    : null;
  const s = engine.feed(input, dt, nowSec);

  // 効果音: acquire に入ったら ready、ロックで captured
  if (
    (s.phase === "acquire" || s.phase === "validate-acquire") &&
    prevPhase !== s.phase &&
    !prevPhase.endsWith("acquire")
  ) {
    beeper.ready();
  }
  prevPhase = s.phase;
  if (s.justLocked) {
    beeper.captured();
  }

  calTarget = s.target;
  calHold = s.holdProgress;
  calPhase = s.phase;
  calDisp = s.dispersion;
  calThresh = s.threshold;
  calGood = s.good;
  calIndex = s.index;
  calTotal = s.total;

  if (s.done && s.model) {
    model = s.model;
    classifier = new EyeContactClassifier(model);
    accuracy = s.accuracy;
    calTarget = null;
    mode = "ACTIVE";
    beeper.complete();
  }
}

function loop(): void {
  if (!running || !tracker || !videoEl || !canvasEl) {
    return;
  }
  const now = performance.now();
  let dt = (now - prev) / 1000;
  prev = now;
  if (dt > 0.1) {
    dt = 0.1;
  }
  if (dt > 0) {
    fps = fps > 0 ? 0.9 * fps + 0.1 * (1 / dt) : 1 / dt;
  }

  if (videoEl.videoWidth && canvasEl.width !== videoEl.videoWidth) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }

  const faceFrame = videoEl.videoWidth ? tracker.process(videoEl, now - startTs) : null;
  faceVisible = faceFrame !== null;
  const analysis = faceFrame ? analyzeFrame(faceFrame) : null;
  if (analysis) {
    frame = {
      gx: analysis.gaze.x,
      gy: analysis.gaze.y,
      yaw: analysis.pose.yaw,
      pitch: analysis.pose.pitch,
    };
  }

  if (mode === "WAIT_FOR_FACE") {
    if (faceVisible) {
      startCalibration();
    }
  } else if (mode === "CALIBRATING") {
    stepCalibration(analysis, now / 1000, dt);
  } else if (mode === "ACTIVE" && classifier && model) {
    if (analysis) {
      ec = classifier.update(analysis.features, analysis.blink, dt);
      const drift = ec.state === ContactState.NoContact ? driftDirection(ec.por, model.camPoint) : null;
      stats.update(dt, ec.state, drift);
      statsView = {
        ratio: stats.contactRatio,
        longest: stats.longestStreak,
        current: stats.currentStreak,
      };
      gazeDot = { x: clamp(ec.por.x, 0.02, 0.98), y: clamp(ec.por.y, 0.02, 0.98) };
    } else {
      stats.update(dt, null);
      ec = null;
      gazeDot = null;
    }
  }

  const ctx = canvasEl.getContext("2d");
  if (ctx && mode !== "CALIBRATING") {
    drawFrame(ctx, videoEl, {
      landmarks: faceFrame?.landmarks ?? null,
      videoW: videoEl.videoWidth || 1,
      videoH: videoEl.videoHeight || 1,
      showLandmarks: showDebug,
    });
  }

  rafId = requestAnimationFrame(loop);
}

function recalibrate(): void {
  beeper.resume();
  startCalibration();
}

function resetStats(): void {
  stats.reset();
  statsView = { ratio: 0, longest: 0, current: 0 };
}

function toggleDebug(): void {
  showDebug = !showDebug;
}

function onKey(e: KeyboardEvent): void {
  beeper.resume();
  if (e.key === "d") {
    toggleDebug();
  } else if (e.key === "c") {
    recalibrate();
  } else if (e.key === "r") {
    resetStats();
  }
}
</script>

<svelte:window onkeydown={onKey} />

<main onpointerdown={() => beeper.resume()}>
  <h1>GazeDetect <small>カメラ目線トレーナー</small></h1>

  {#if errorMsg}
    <p class="error">{errorMsg}</p>
  {/if}

  <div class="stage {contactClass}">
    <!-- svelte-ignore a11y_media_has_caption -->
    <video bind:this={videoEl} playsinline muted hidden></video>
    <canvas bind:this={canvasEl}></canvas>
    {#if mode === "ACTIVE"}
      <Hud {mode} {faceVisible} {ec} {showDebug} {fps} {frame} stats={statsView} />
    {/if}
  </div>

  {#if mode === "ACTIVE" && accuracy !== null}
    <p class="accuracy">キャリブ精度（held-out）: ズレ {(accuracy * 100).toFixed(1)}%（小さいほど良い）</p>
  {/if}

  <Controls onCalibrate={recalibrate} onToggleDebug={toggleDebug} onReset={resetStats} {showDebug} />
</main>

<!-- 予測視線点（ウィンドウ全体座標） -->
{#if mode === "ACTIVE" && gazeDot}
  <div class="gaze-dot viewport" class:bright={showDebug} style="left:{gazeDot.x * 100}%; top:{gazeDot.y * 100}%"></div>
{/if}

<!-- キャリブは全ウィンドウ（外挿を短く＝カメラ位置整合を改善） -->
{#if mode === "CALIBRATING"}
  <div class="cal-fullscreen">
    {#if calIsCamera}
      <!-- カメラ注視は画面外を見る → 画面に位置マーカーを置かない（大きな文字のみ） -->
      <div class="cal-lens">
        <div class="arrow">↑</div>
        <div class="big">カメラのレンズを見てください</div>
        <div class="sub2">
          {#if !faceVisible}顔が見えません（明るく・正面に）{:else if calGood}そのまま見続けて…{:else}画面ではなく、上にある小さなレンズを{/if}
        </div>
      </div>
      <div class="cal-progress"><div class="bar" style="width:{calHold * 100}%"></div></div>
    {:else}
      <div class="cal-instruction">
        {#if calKind === "validate"}確認です。{/if}{calTarget?.label ?? ""}を見つめてください
        <span class="sub">
          {#if !faceVisible}顔が見えません（明るく・正面に）{:else}しっかり見続けるとリングが満ちます（頭は少し動かしてOK・ウィンドウは大きめ推奨）{/if}
          　{calIndex + 1}/{calTotal}
        </span>
      </div>
      {#if calTarget}
        <div
          class="cal-target {calKind} {calPhase} {calGood ? 'good' : ''}"
          style="left:{calPos.x}%; top:{calPos.y}%; --pct:{calHold * 100}%"
        >
          <div class="ring"></div>
          <div class="dot"></div>
        </div>
      {/if}
    {/if}
    {#if showDebug}
      <div class="cal-debug">
        disp {calDisp === Number.POSITIVE_INFINITY ? "∞" : calDisp.toFixed(3)}
        thr {calThresh === Number.POSITIVE_INFINITY ? "∞" : calThresh.toFixed(3)}
        good {calGood ? "✓" : "—"}　hold {(calHold * 100).toFixed(0)}%　{calPhase}
      </div>
    {/if}
  </div>
{/if}
