import { join } from "path";

const root = join(import.meta.dir, "..");
const bake = join(root, "native/bridge-build/ryugu_gradient_bake_cpp");
const obj =
  Bun.env.BAKE_OBJ ??
  join(root, "../Ryugu_wasm/assets/models/SHAPE_SFM_200k_v20180804.obj");
const out = join(root, "assets/gradient_faces.bin");

const r = Bun.spawnSync([bake, "--obj", obj, "--out", out], {
  cwd: root,
  stdout: "inherit",
  stderr: "inherit",
});
if (r.exitCode !== 0) process.exit(r.exitCode ?? 1);
console.log(`bake → ${out}`);
