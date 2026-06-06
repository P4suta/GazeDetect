import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// https://vite.dev/config/
// GitHub Pages（プロジェクトページ）はサブパス配信なので、ビルド時は相対ベースにする。
export default defineConfig(({ command }) => ({
  base: command === "build" ? "./" : "/",
  plugins: [svelte()],
}));
