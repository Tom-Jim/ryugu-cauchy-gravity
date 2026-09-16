import { mkdirSync } from "fs";
import { join } from "path";

const root = join(import.meta.dir, "..");
const profile = Bun.argv.includes("--dev") ? "dev" : "release";
const glb = join(root, "assets/models/ryugu.glb");
const wasm = join(root, "pkg/ryugu_cauchy_gravity_bg.wasm");

if (profile === "release") {
  const probe = Bun.spawnSync(["wasm-opt", "--version"], {
    stdout: "ignore",
    stderr: "ignore",
  });
  if (probe.exitCode !== 0) {
    console.error("wasm-opt is required for release builds (install Binaryen)");
    process.exit(1);
  }
}

mkdirSync(join(root, "assets/models"), { recursive: true });

if (!(await Bun.file(glb).exists())) {
  const src = Bun.file(join(root, "../Ryugu_wasm/assets/models/ryugu.glb"));
  if (!(await src.exists())) {
    console.error("Missing assets/models/ryugu.glb");
    process.exit(1);
  }
  await Bun.write(glb, src);
}

const r = Bun.spawnSync(
  [
    "wasm-pack",
    "build",
    `--${profile}`,
    "--target",
    "web",
    "--out-dir",
    "pkg",
    "--out-name",
    "ryugu_cauchy_gravity",
  ],
  { cwd: root, stdout: "inherit", stderr: "inherit", env: { ...process.env, RUSTC_WRAPPER: "" } },
);
if (r.exitCode !== 0) process.exit(r.exitCode ?? 1);
if (profile === "release") {
  const size = Bun.file(wasm).size;
  const sizeMiB = size / 1024 / 1024;
  if (size > 40 * 1024 * 1024) {
    console.error(`release WASM is ${sizeMiB.toFixed(1)} MiB; budget is 40.0 MiB`);
    process.exit(1);
  }
  console.log(`ok -> pkg/ (${profile}, WASM ${sizeMiB.toFixed(1)} MiB)`);
} else {
  console.log(`ok -> pkg/ (${profile})`);
}
