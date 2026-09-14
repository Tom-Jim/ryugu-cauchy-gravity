import { mkdirSync } from "fs";
import { join } from "path";

const root = join(import.meta.dir, "..");
const profile = Bun.argv.includes("--dev") ? "dev" : "release";

mkdirSync(join(root, "assets/models"), { recursive: true });

const glbSrc = join(root, "models/ryugu.glb");
const glbDst = join(root, "assets/models/ryugu.glb");
const srcFile = Bun.file(glbSrc);
if (!(await srcFile.exists())) {
  console.error("Missing models/ryugu.glb");
  process.exit(1);
}
if (!(await Bun.file(glbDst).exists())) {
  await Bun.write(glbDst, srcFile);
  console.log("Copied assets/models/ryugu.glb");
}

const result = Bun.spawnSync(
  [
    "wasm-pack",
    "build",
    `--${profile}`,
    "--target",
    "web",
    "--out-dir",
    "pkg",
    "--out-name",
    "ryugu_h_cal",
  ],
  {
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
    env: { ...process.env, RUSTC_WRAPPER: "" },
  },
);

if (result.exitCode !== 0) {
  process.exit(result.exitCode ?? 1);
}

console.log(`WASM build (${profile}) → pkg/`);
