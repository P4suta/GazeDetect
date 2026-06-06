# GazeDetect

[![ci](https://github.com/P4suta/GazeDetect/actions/workflows/ci.yml/badge.svg)](https://github.com/P4suta/GazeDetect/actions/workflows/ci.yml)
[![deploy](https://github.com/P4suta/GazeDetect/actions/workflows/deploy.yml/badge.svg)](https://github.com/P4suta/GazeDetect/actions/workflows/deploy.yml)

ブラウザ完結で「カメラ目線（アイコンタクト）」をリアルタイム検知する面接練習アプリ。
レンズを見ているかを OK / NG 表示し、目線維持率を集計する。

**▶ デモ: https://p4suta.github.io/GazeDetect/** （Chromium 系推奨・カメラ許可が必要。ウィンドウは大きめに）

## 仕組み

カメラは画面の上にあり、画面を見ると少し下を向く＝相手からは目線が外れる。本当の「カメラ目線」は
レンズを見ること——これを訓練する。

- **頭部分離**: 頭部姿勢(MediaPipe の 4×4 顔変換行列)で de-rotate した **eye-in-head（眼球内）視線**を
  主特徴にし、頭の動きと目の動きを分離。
- **キャリブは品質で勝ち取る**: 適応しきい値以下の固視を“連続して”保持できたときだけ点が確定。崩れたら
  猶予つきで減衰（リングは品質を正直に表す）。学習は座標既知の画面グリッド点のみ、**カメラ点は「レンズ
  注視」視線の外挿で実測**（機種・配置に整合）。学習後に held-out 点で精度を検証。
- **判定**: リッジ回帰で画面上の注視点(PoR)を推定 → One-Euro 平滑化 → カメラ点との距離で二重しきい値
  ヒステリシス、瞬きは凍結。

技術スタック: **Bun + Vite + Svelte 5 + Biome + bun test**（ツールは [mise](https://mise.jdx.dev/) 管理）、
推論は **MediaPipe Tasks (FaceLandmarker, WASM)**。すべて client-side（サーバへ映像を送らない）。

## クイックスタート（Web アプリ ＝ 本体）

```bash
cd web
mise install      # bun を用意（mise.toml）
mise run install  # 依存
mise run dev      # 開発サーバ → ブラウザでカメラ許可
```

詳細・調整は [`web/README.md`](web/README.md)。

## リポジトリ構成

| パス | 内容 |
| --- | --- |
| `web/` | **本体**: ブラウザ完結の Web アプリ（Vite + Svelte 5 + TypeScript） |
| ルートの Python (`main.py`/`gaze.py` 他) | 原型のデスクトップ版（MediaPipe + OpenCV）。ロジックの参照用に残置 |
| `.github/workflows/` | CI（typecheck/lint/test/build）と GitHub Pages デプロイ |

## ライセンス

[MIT](LICENSE)
