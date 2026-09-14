import { join } from "path";

const PORT = Number.parseInt(Bun.env.PORT ?? "3000", 10);
const ROOT = join(import.meta.dir, "..");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".bin": "application/octet-stream",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
};

const server = Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    const relativePath =
      pathname === "/index.html" ? "src/index.html" : pathname.slice(1);
    const file = Bun.file(join(ROOT, relativePath));
    if (!(await file.exists())) {
      return new Response("Not Found", { status: 404 });
    }
    const ext = pathname.slice(pathname.lastIndexOf("."));
    return new Response(file, {
      headers: {
        "Content-Type": MIME[ext] ?? "application/octet-stream",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cache-Control": "no-store",
      },
    });
  },
});

console.log(`Ryugu viewer at http://127.0.0.1:${PORT}`);
