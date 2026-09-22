import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const root = join(import.meta.dir, "..", "..", "..");
const out = join(root, "dist");

/** Only ship files the browser actually needs. */
const files = [
  "pkg/ryugu_cauchy_gravity.js",
  "pkg/ryugu_cauchy_gravity_bg.wasm",
  "pkg/app.js",
  "assets/models/ryugu.glb",
  "assets/models/Deimos.glb",
  "assets/models/Phobos.glb",
  "assets/density/cauchy_elliptic.toml",
];

// Keep every checked-in model available to the same asset library used by the
// dev server. This also makes `pages:stage` correct when a new GLB is added.
for (const name of readdirSync(join(root, "assets/models"))) {
  if (name.toLowerCase().endsWith(".glb")) {
    const entry = `assets/models/${name}`;
    if (!files.includes(entry)) files.push(entry);
  }
}

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

for (const file of files) {
  const target = join(out, file);
  mkdirSync(dirname(target), { recursive: true });
  cpSync(join(root, file), target);
}

const sourceHtml = readFileSync(join(root, "src/web/index.html"), "utf8");
// The viewer is static-only now (the server never computes), so the shipped
// page already carries the static marker. Assert it rather than rewriting it,
// which would silently mask a page left in a stale mode.
if (!sourceHtml.includes('<meta name="ryugu-mode" content="static" />')) {
  throw new Error("src/web/index.html is missing its static-mode marker");
}
writeFileSync(join(out, "index.html"), sourceHtml);
writeFileSync(join(out, ".nojekyll"), "");

console.log("ok -> dist/ (GitHub Pages)");
