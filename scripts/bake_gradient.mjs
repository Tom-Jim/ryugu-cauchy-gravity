import { join } from "path";

const root = join(import.meta.dir, "..");
const bake = join(root, "native/bridge-build/ryugu_gradient_bake_cpp");
const obj = join(root, "models/ryugu.obj");
const out = join(root, "assets/gradient_faces.bin");

const stride = Bun.env.BAKE_STRIDE ?? "96";
const maxFaces = Bun.env.BAKE_MAX_FACES ?? "2048";

const result = Bun.spawnSync(
  [bake, "--obj", obj, "--out", out, "--stride", stride, "--max-faces", maxFaces],
  { cwd: root, stdout: "inherit", stderr: "inherit" },
);
if (result.exitCode !== 0) {
  console.error("Bake failed. Build the native bridge first: bun run esa:bridge");
  process.exit(result.exitCode ?? 1);
}
console.log(`Bake complete → ${out}`);
console.log("Rebuild the WASM package afterwards so include_bytes picks up the new file.");
