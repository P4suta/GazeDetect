// canvas 描画ヘルパ。ミラー表示した映像＋（デバッグ時）虹彩・目のランドマーク点を描く。
// キャリブのターゲットや視線(PoR)ドットは画面座標系なので App 側の DOM 重畳で描く。

import type { Point } from "./features";

const IRIS = [
  [468, 469, 470, 471, 472],
  [473, 474, 475, 476, 477],
];
const EYE_CORNERS = [33, 133, 362, 263];

export interface DrawOptions {
  landmarks: Point[] | null;
  videoW: number;
  videoH: number;
  showLandmarks: boolean;
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  options: DrawOptions,
): void {
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1); // ミラー表示
  ctx.drawImage(video, 0, 0, width, height);
  if (options.landmarks && options.showLandmarks) {
    drawLandmarks(ctx, options.landmarks, width / options.videoW, height / options.videoH);
  }
  ctx.restore();
}

function drawLandmarks(ctx: CanvasRenderingContext2D, pts: Point[], sx: number, sy: number): void {
  ctx.fillStyle = "#00c800";
  for (const ring of IRIS) {
    let cx = 0;
    let cy = 0;
    for (const i of ring) {
      cx += pts[i].x;
      cy += pts[i].y;
    }
    ctx.beginPath();
    ctx.arc((cx / ring.length) * sx, (cy / ring.length) * sy, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#00c8ff";
  for (const i of EYE_CORNERS) {
    ctx.beginPath();
    ctx.arc(pts[i].x * sx, pts[i].y * sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}
