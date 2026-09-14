import { mkdirSync } from "fs";
import { join } from "path";

const root = join(import.meta.dir, "..");
const profile = Bun.argv.includes("--dev") ? "dev" : "release";
const glb = join(root, "assets/models/ryugu.glb");

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
  ["wasm-pack", "build", `--${profile}`, "--target", "web", "--out-dir", "pkg", "--out-name", "ryugu_h_cal"],
  { cwd: root, stdout: "inherit", stderr: "inherit", env: { ...process.env, RUSTC_WRAPPER: "" } },
);
if (r.exitCode !== 0) process.exit(r.exitCode ?? 1);
console.log(`ok → pkg/ (${profile})`);
