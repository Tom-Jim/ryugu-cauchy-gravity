import { join } from "path";

const root = join(import.meta.dir, "..", "..", "..");
const profile = Bun.argv.includes("--dev") ? "dev" : "release";
const glb = join(root, "assets/models/ryugu.glb");
const wasm = join(root, "pkg/ryugu_cauchy_gravity_bg.wasm");
const vueCompiler = join(root, "node_modules/vue/dist/vue.esm-bundler.js");

function readU32(bytes, cursor) {
  let value = 0;
  let shift = 0;
  while (true) {
    const byte = bytes[cursor.offset++];
    value |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return value >>> 0;
    shift += 7;
  }
}

function assertExternrefTable(bytes) {
  const cursor = { offset: 8 };
  const tables = [];
  let externrefExport = null;
  const decoder = new TextDecoder();

  while (cursor.offset < bytes.length) {
    const sectionId = bytes[cursor.offset++];
    const size = readU32(bytes, cursor);
    const end = cursor.offset + size;
    if (sectionId === 4) {
      const count = readU32(bytes, cursor);
      for (let index = 0; index < count; index++) {
        const refType = bytes[cursor.offset++];
        const flags = bytes[cursor.offset++];
        const initial = readU32(bytes, cursor);
        const maximum = flags & 1 ? readU32(bytes, cursor) : null;
        tables.push({ refType, initial, maximum });
      }
    } else if (sectionId === 7) {
      const count = readU32(bytes, cursor);
      for (let index = 0; index < count; index++) {
        const nameLength = readU32(bytes, cursor);
        const name = decoder.decode(
          bytes.subarray(cursor.offset, cursor.offset + nameLength),
        );
        cursor.offset += nameLength;
        const kind = bytes[cursor.offset++];
        const itemIndex = readU32(bytes, cursor);
        if (name === "__wbindgen_externrefs") {
          externrefExport = { kind, itemIndex };
        }
      }
    }
    cursor.offset = end;
  }

  const table = externrefExport?.kind === 1
    ? tables[externrefExport.itemIndex]
    : null;
  if (table?.refType !== 0x6f || table.maximum !== null) {
    throw new Error(
      "release WASM has an invalid __wbindgen_externrefs table; "
      + "check the Binaryen version used by wasm-opt",
    );
  }
}

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

if (!(await Bun.file(glb).exists())) {
  console.error("Missing assets/models/ryugu.glb");
  process.exit(1);
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
const bundle = await Bun.build({
  entrypoints: [join(root, "src/web/app.ts")],
  outdir: join(root, "pkg"),
  target: "browser",
  format: "esm",
  naming: "app.js",
  define: {
    __VUE_OPTIONS_API__: "true",
    __VUE_PROD_DEVTOOLS__: "false",
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: "false",
  },
  // `app.ts` mounts the template already present in index.html. Vue's default
  // bundler export is runtime-only and clears that template without compiling
  // it, leaving the Bevy canvas visible but every control missing.
  plugins: [{
    name: "vue-runtime-compiler",
    setup(build) {
      build.onResolve({ filter: /^vue$/ }, () => ({ path: vueCompiler }));
    },
  }],
  minify: profile === "release",
  sourcemap: profile === "dev" ? "external" : "none",
});
if (!bundle.success) {
  for (const log of bundle.logs) console.error(log);
  process.exit(1);
}
if (profile === "release") {
  const wasmBytes = new Uint8Array(await Bun.file(wasm).arrayBuffer());
  assertExternrefTable(wasmBytes);
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
