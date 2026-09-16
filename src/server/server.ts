/**
 * Development server: static asset host plus one HTTP control surface per solver.
 *
 * The server deliberately owns almost no solver logic. Each algorithm lives in
 * `./solvers/<name>.ts`, exports the same three-entry contract (`boot`, `status`,
 * `start`) and is registered in the `SOLVERS` table below. Adding an algorithm
 * is then a two-line change here plus one file there, and the routing layer
 * never has to know whether the solver is a native binary, a WASM/WebGPU bake or
 * something else entirely.
 *
 * Routes:
 *   GET  /api/<solver>/status   current bake state, progress and record summary
 *   POST /api/<solver>/start    { mode: "restart" | "resume", standoffMm, density }
 *   GET  /*                     static files from the repository root
 */
import { join, normalize, relative } from "node:path";
import { bootCarlsonStatus, carlsonStatusResponse, startCarlsonBake } from "./solvers/carlson";
import {
  bootCarlsonAlphaStatus,
  carlsonalphaStatusResponse,
  startCarlsonAlphaBake,
} from "./solvers/carlsonalpha";
import { corsJson } from "./solvers/gpu_solver";
import { bootMasconStatus, masconStatusResponse, startMasconBake } from "./solvers/mascon";
import { bootRtfpStatus, rtfpStatusResponse, startRtfpBake } from "./solvers/rtfp";
import { bootWernerStatus, startWernerBake, wernerStatusResponse } from "./solvers/werner";
import { liveRecordPath, type LiveSolver } from "./solvers/live_paths";

const ROOT = join(import.meta.dir, "../..");
const PORT = Number(Bun.env.PORT ?? 3000);

/** Everything the viewer is allowed to load, and nothing else. */
const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".obj": "text/plain; charset=utf-8",
  ".bin": "application/octet-stream",
  ".wgsl": "text/plain; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".css": "text/css; charset=utf-8",
};

type StartMode = "restart" | "resume";

type StartArgs = {
  mode: StartMode;
  standoffMm?: number;
  density?: string;
};

type SolverRoute = {
  boot: () => void;
  status: () => Response | Promise<Response>;
  start: (args: StartArgs) => Response | Promise<Response>;
};

/**
 * One entry per algorithm. Werner ignores `density`; the other routes use it to
 * select the density model and therefore the record they own.
 */
const SOLVERS: Record<string, SolverRoute> = {
  "/api/werner": {
    boot: bootWernerStatus,
    status: wernerStatusResponse,
    start: ({ mode, standoffMm }) => startWernerBake(mode, standoffMm),
  },
  "/api/mascon": {
    boot: bootMasconStatus,
    status: masconStatusResponse,
    start: ({ mode, standoffMm, density }) => startMasconBake(mode, standoffMm, density),
  },
  "/api/rtfp": {
    boot: bootRtfpStatus,
    status: rtfpStatusResponse,
    start: ({ mode, standoffMm, density }) => startRtfpBake(mode, standoffMm, density),
  },
  "/api/carlson": {
    boot: bootCarlsonStatus,
    status: carlsonStatusResponse,
    start: ({ mode, standoffMm, density }) => startCarlsonBake(mode, standoffMm, density),
  },
  "/api/carlsonalpha": {
    boot: bootCarlsonAlphaStatus,
    status: carlsonalphaStatusResponse,
    start: ({ mode, standoffMm, density }) => startCarlsonAlphaBake(mode, standoffMm, density),
  },
};

/** A malformed or absent body means "restart at the server's current height". */
async function readStartArgs(req: Request): Promise<StartArgs> {
  try {
    const body = (await req.json()) as {
      mode?: string;
      standoffMm?: number;
      density?: string;
    };
    return {
      mode: body?.mode === "resume" ? "resume" : "restart",
      standoffMm: Number.isFinite(body?.standoffMm) ? body?.standoffMm : undefined,
      density: typeof body?.density === "string" ? body.density : undefined,
    };
  } catch {
    return { mode: "restart" };
  }
}

/**
 * Resolve a request path to a file inside the repository, or `null`.
 *
 * `relative()` is the containment check: anything that escapes ROOT (via `..`,
 * an absolute path, or a symlinked prefix) produces a leading `..` and is
 * rejected rather than served.
 */
function resolveStatic(pathname: string): string | null {
  const requested = pathname === "/" ? "src/web/index.html" : pathname.slice(1);
  const target = normalize(join(ROOT, requested));
  const rel = relative(ROOT, target);
  if (rel.startsWith("..") || rel === "") return null;
  return target;
}

/** Shared isolation headers; the viewer needs cross-origin isolation for WASM. */
function staticHeaders(ext: string): Record<string, string> {
  return {
    "Content-Type": MIME[ext] ?? "application/octet-stream",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
  };
}

Bun.serve({
  port: PORT,
  hostname: "0.0.0.0",
  async fetch(req) {
    const { pathname, searchParams } = new URL(req.url);

    const recordMatch = pathname.match(
      /^\/api\/(werner|mascon|rtfp|carlson|carlsonalpha)\/record$/,
    );
    if (recordMatch) {
      const solver = recordMatch[1] as LiveSolver;
      const defaultDensity = solver === "werner"
        ? "uniform"
        : solver === "mascon"
          ? "elliptic"
          : solver === "rtfp" || solver === "carlson"
            ? "cauchy"
            : "elliptic";
      const density = searchParams.get("density") ?? defaultDensity;
      try {
        const file = liveRecordPath(solver, density);
        if (!(await Bun.file(file).exists())) {
          return new Response("Not Found", { status: 404 });
        }
        return new Response(Bun.file(file), {
          headers: {
            "Content-Type": "application/octet-stream",
            "Cache-Control": "no-store",
            "Cross-Origin-Resource-Policy": "same-origin",
          },
        });
      } catch (error) {
        return new Response(String(error), { status: 400 });
      }
    }

    const solver = SOLVERS[pathname.replace(/\/(status|start)$/, "")];
    if (solver) {
      if (pathname.endsWith("/status")) return solver.status();
      if (pathname.endsWith("/start") && req.method === "POST") {
        return solver.start(await readStartArgs(req));
      }
      return corsJson({ error: `use GET ${pathname}/status or POST ${pathname}/start` }, 405);
    }

    const file = resolveStatic(pathname);
    if (file === null || !(await Bun.file(file).exists())) {
      return new Response("Not Found", { status: 404 });
    }
    const ext = pathname === "/" ? ".html" : pathname.slice(pathname.lastIndexOf("."));
    return new Response(Bun.file(file), { headers: staticHeaders(ext) });
  },
});

console.log(`http://127.0.0.1:${PORT}`);

// Boot every solver so a fresh server describes the records already on disk
// instead of reporting "not started" until the first poll.
for (const solver of Object.values(SOLVERS)) {
  try {
    solver.boot();
  } catch (error) {
    console.error("solver boot failed:", error);
  }
}
