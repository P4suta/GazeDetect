# GazeDetect (Web)

ブラウザ完結のカメラ目線（アイコンタクト）検知。getUserMedia + MediaPipe Web (FaceLandmarker)。

## 仕組み

**視線推定**: 頭の動きと目の動きを分離するため、頭部姿勢(4×4 顔変換行列)で de-rotate した
**eye-in-head（眼球内）視線**を主特徴にする（画像座標の虹彩比率は使わない）。これに blendshape の
eyeLook 群と頭部姿勢/位置を加え、リッジ回帰で画面上の注視点(PoR)へ写像。**学習は座標既知の画面
グリッド点のみ**で行い、**カメラ点はハードコードせず「レンズ注視」の視線を学習済みモデルで外挿して
実測**する（機種・配置に依らずカメラ位置と整合）。「カメラ目線」＝予測 PoR がその camPoint の近傍
（One-Euro 平滑化＋二重しきい値ヒステリシス、瞬きは凍結）。

**キャリブ（“OK は勝ち取る”）**: 各点は **適応しきい値以下の固視を“連続して”一定時間保持** できた
ときだけ lock する。しきい値は固定値ではなく **ユーザー自身の固視ばらつきの分位点に自動追従**
（カメラ/照明に依らず初回から機能・絶対ロックしない問題を回避）。崩れたら **猶予つきで減衰**
（瞬きでは台無しにしない）ので、リングは品質を**正直に**表す（しっかり見ると満ち、外すと減る）。
時間ベースの junk 受理は廃止。学習後に **held-out（学習に使わない）点で精度を検証**して表示する。

純粋ロジックは DOM 非依存で bun test 可能（linalg/headpose/filter/fixation/calibengine/gaze/calibration/stats）。

## ツール / コマンド

Bun（mise 管理）+ Vite + Svelte 5 + Biome + bun test。

```bash
cd web
mise install        # bun を用意（初回のみ）
mise run install    # 依存インストール
mise run dev        # 開発サーバ → ブラウザでカメラ許可
mise run build      # dist/ に静的ビルド
mise run typecheck  # svelte-check
mise run lint       # biome
mise run test       # 純粋ロジックのテスト
```

bun を直接叩くなら shell で `mise activate`。無ければ `mise run <task>` か `mise exec -- bun ...`。

## 使い方

`mise run dev` → カメラ許可 → 画面が白くなり、青い点を見つめる。**しっかり見続けるとリングが満ち**、
満ちると緑＋音で確定。9点＋カメラのレンズ、最後に紫の「確認の点」（精度検証）。完了で
**held-out 精度（ズレ%）** を表示。操作はボタン/キー（`c` 再キャリブ / `d` デバッグ / `r` リセット）。

## 精度の確認・調整（実機前提）

`d` でデバッグ表示。

- **頭部分離の受け入れテスト**: カメラを固視して頭を ±20° 振る。`eye(gx,gy)` がほぼ動かなければ
  OK。頭に追従するなら座標規約差なので `src/lib/features.ts` の `GAZE_SIGN`（まず `y:-1`）を切替。
- **キャリブの手応え**: デバッグの `disp/thr/good/hold` を見る。なかなか満ちないなら
  `src/lib/calibengine.ts` の `DEFAULT_ENGINE_CONFIG`（`percentile` を上げる＝緩い、`hold` を短く）。
- **判定**: `src/lib/gaze.ts` の `DEFAULT_EC_CONFIG`（`enterR`/`exitR`/`minCutoff`/`beta`）。

> 注: 頭部分離・検知精度の最終確認はカメラ実機が必要（ヘッドレス不可）。純粋ロジック（適応しきい値・
> 連続固視ロック・減衰・回帰・held-out 検証・判定）は単体テストで担保。座標規約と体感は実機で詰める。
