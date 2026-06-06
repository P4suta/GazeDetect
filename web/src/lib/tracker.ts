// MediaPipe Web FaceLandmarker のラッパ。
// 3D ランドマーク（z は x とほぼ同スケール）＋ blendshapes ＋ 顔変換行列を返す。顔が無ければ null。

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { FaceFrame } from "./features";
import { IDENTITY16 } from "./headpose";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export class FaceTracker {
  private lastTs = -1;

  private constructor(private readonly landmarker: FaceLandmarker) {}

  static async create(): Promise<FaceTracker> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const make = (delegate: "GPU" | "CPU"): Promise<FaceLandmarker> =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
    try {
      return new FaceTracker(await make("GPU"));
    } catch {
      return new FaceTracker(await make("CPU"));
    }
  }

  process(video: HTMLVideoElement, timestampMs: number): FaceFrame | null {
    const ts = Math.max(Math.round(timestampMs), this.lastTs + 1);
    this.lastTs = ts;
    const result = this.landmarker.detectForVideo(video, ts);
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) {
      return null;
    }
    const w = video.videoWidth;
    const h = video.videoHeight;
    // z は MediaPipe 規約で x とほぼ同スケール → z*w でピクセル系に揃える。
    const landmarks = faces[0].map((p) => ({ x: p.x * w, y: p.y * h, z: p.z * w }));

    const blend: Record<string, number> = {};
    const categories = result.faceBlendshapes?.[0]?.categories;
    if (categories) {
      for (const c of categories) {
        blend[c.categoryName] = c.score;
      }
    }

    const matrix = result.facialTransformationMatrixes?.[0]?.data ?? IDENTITY16;
    return { landmarks, blend, matrix: Array.from(matrix) };
  }

  close(): void {
    this.landmarker.close();
  }
}
