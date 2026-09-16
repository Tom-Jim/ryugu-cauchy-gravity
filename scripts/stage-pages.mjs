import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const root = join(import.meta.dir, "..");
const out = join(root, "dist");

/** Only ship files the browser actually needs. */
const files = [
  "pkg/ryugu_cauchy_gravity.js",
  "pkg/ryugu_cauchy_gravity_bg.wasm",
  "src/web/live_preview.js",
  "assets/models/ryugu.glb",
  "assets/live/face_observers.bin",
  "assets/live/preview_sources.bin",
];

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

for (const file of files) {
  const target = join(out, file);
  mkdirSync(dirname(target), { recursive: true });
  cpSync(join(root, file), target);
}

const sourceHtml = readFileSync(join(root, "src/web/index.html"), "utf8");
const html = sourceHtml.replace(
  '<meta name="ryugu-mode" content="dynamic" />',
  '<meta name="ryugu-mode" content="static" />',
);
if (html === sourceHtml) {
  throw new Error("failed to switch src/web/index.html into static demo mode");
}
writeFileSync(join(out, "index.html"), html);
writeFileSync(join(out, ".nojekyll"), "");

console.log("ok -> dist/ (GitHub Pages)");
