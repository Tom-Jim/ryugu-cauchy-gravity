import { extname, join } from "node:path";

const ROOT = join(import.meta.dir, "../../..");
const PUBLIC_FILES = new Map([
  ["/", "src/web/index.html"],
  ["/pkg/app.js", "pkg/app.js"],
  ["/pkg/ryugu_cauchy_gravity.js", "pkg/ryugu_cauchy_gravity.js"],
  ["/pkg/ryugu_cauchy_gravity_bg.wasm", "pkg/ryugu_cauchy_gravity_bg.wasm"],
  ["/assets/models/ryugu.glb", "assets/models/ryugu.glb"],
  ["/assets/models/Deimos.glb", "assets/models/Deimos.glb"],
  ["/assets/models/Phobos.glb", "assets/models/Phobos.glb"],
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

/** Serve only the files needed by the static browser runtime. */
export async function serveStatic(req: Request): Promise<Response> {
  const pathname = new URL(req.url).pathname;
  // Keep the explicit allow-list for the runtime, but resolve model/density
  // assets from the checked-in assets tree as a second path. This prevents a
  // newly added GLB from becoming a browser-visible 404 merely because the
  // allow-list or a stale package omitted one entry.
  const relative = PUBLIC_FILES.get(pathname)
    ?? (/^\/assets\/(models|density)\/[A-Za-z0-9_.-]+\.(glb|toml)$/.test(pathname)
      ? pathname.slice(1)
      : undefined);
  if (relative === undefined) {
    return new Response("Not Found", { status: 404 });
  }
  // Development serves from the repository root; packaged/static hosts may
  // serve from `dist`. Try both so an imported GLB never turns into a false
  // 404 merely because the host selected the packaged tree.
  const rootPath = join(ROOT, relative);
  const candidates = [rootPath, join(ROOT, "dist", relative)];
  let path = rootPath;
  let file = Bun.file(path);
  for (const candidate of candidates) {
    const candidateFile = Bun.file(candidate);
    if (await candidateFile.exists()) {
      path = candidate;
      file = candidateFile;
      break;
    }
  }
  if (!(await file.exists())) {
    return new Response("Not Found", { status: 404 });
  }
  return new Response(file, { headers: headers(path) });
}
