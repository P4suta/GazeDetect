<script lang="ts">
interface Props {
  onCalibrate: () => void;
  onToggleDebug: () => void;
  onReset: () => void;
  showDebug: boolean;
}

let { onCalibrate, onToggleDebug, onReset, showDebug }: Props = $props();
</script>

<div class="controls">
  <button type="button" onclick={onCalibrate}>再キャリブ (c)</button>
  <button type="button" onclick={onToggleDebug}>
    {showDebug ? "デバッグ非表示" : "デバッグ表示"} (d)
  </button>
  <button type="button" onclick={onReset}>統計リセット (r)</button>
</div>
