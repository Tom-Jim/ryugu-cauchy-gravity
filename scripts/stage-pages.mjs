import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const root = join(import.meta.dir, "..");
const out = join(root, "dist");

/** Only ship files the browser actually needs. */
const files = [
  "pkg/ryugu_cauchy_gravity.js",
  "pkg/ryugu_cauchy_gravity_bg.wasm",
  "assets/models/ryugu.glb",
  "assets/records/gradient_faces.bin",
  "assets/records/mascon_faces.bin",
  "assets/records/mascon_elliptic_faces.bin",
  "assets/records/rtfp_faces.bin",
  "assets/records/rtfp_constant_faces.bin",
  "assets/records/carlson_cauchy_faces.bin",
  "assets/records/carlson_constant_faces.bin",
  "assets/records/carlsonalpha_elliptic_faces.bin",
  "assets/records/carlsonalpha_constant_faces.bin",
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
