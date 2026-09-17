import { extname, join } from "node:path";

const ROOT = join(import.meta.dir, "../../..");
const PUBLIC_FILES = new Map([
  ["/", "src/web/index.html"],
  ["/pkg/app.js", "pkg/app.js"],
  ["/pkg/ryugu_cauchy_gravity.js", "pkg/ryugu_cauchy_gravity.js"],
  ["/pkg/ryugu_cauchy_gravity_bg.wasm", "pkg/ryugu_cauchy_gravity_bg.wasm"],
  ["/assets/models/ryugu.glb", "assets/models/ryugu.glb"],
  ["/assets/density/cauchy.toml", "assets/density/cauchy.toml"],
  ["/assets/density/cauchy_elliptic.toml", "assets/density/cauchy_elliptic.toml"],
]);

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".toml": "text/plain; charset=utf-8",
};

function headers(path: string): Record<string, string> {
  return {
    "Content-Type": MIME[extname(path)] ?? "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
  };
}

/** Serve only the seven files needed by the static browser runtime. */
export async function serveStatic(req: Request): Promise<Response> {
  const pathname = new URL(req.url).pathname;
  const relative = PUBLIC_FILES.get(pathname);
  if (relative === undefined) {
    return new Response("Not Found", { status: 404 });
  }
  const path = join(ROOT, relative);
  const file = Bun.file(path);
  if (!(await file.exists())) {
    return new Response("Not Found", { status: 404 });
  }
  return new Response(file, { headers: headers(path) });
}
