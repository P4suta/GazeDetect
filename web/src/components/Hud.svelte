<script lang="ts">
import { ContactState, type EyeContactDebug } from "../lib/gaze";

interface Props {
  mode: string;
  faceVisible: boolean;
  ec: EyeContactDebug | null;
  showDebug: boolean;
  fps: number;
  frame: { gx: number; gy: number; yaw: number; pitch: number } | null;
  stats: { ratio: number; longest: number; current: number };
}

let { mode, faceVisible, ec, showDebug, fps, frame, stats }: Props = $props();

const DEG = 180 / Math.PI;

const status = $derived.by(() => {
  if (mode === "WAIT_FOR_FACE") {
    return "顔を画面に入れてください";
  }
  if (!faceVisible || !ec) {
    return "顔を見失いました";
  }
  return ec.state === ContactState.Contact ? "◎ カメラ目線 OK" : "✕ 目線をカメラへ";
});
</script>

<div class="hud">
  <div class="status">{status}</div>
  {#if mode === "ACTIVE"}
    <div class="metrics">
      維持率 {(stats.ratio * 100).toFixed(0)}%　最長 {stats.longest.toFixed(1)}s　現在 {stats.current.toFixed(1)}s
    </div>
  {/if}
  {#if showDebug}
    <div class="debug">
      {#if ec}
        <div>dist {ec.distance.toFixed(3)}　blink {ec.blink ? "Y" : "N"}　fps {fps.toFixed(0)}</div>
        <div>por ({ec.por.x.toFixed(2)}, {ec.por.y.toFixed(2)})　raw ({ec.rawPor.x.toFixed(2)}, {ec.rawPor.y.toFixed(2)})</div>
      {/if}
      {#if frame}
        <div>eye ({frame.gx.toFixed(2)}, {frame.gy.toFixed(2)})　head yaw {(frame.yaw * DEG).toFixed(0)}° pitch {(frame.pitch * DEG).toFixed(0)}°</div>
      {/if}
    </div>
  {/if}
</div>
