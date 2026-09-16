import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * Runtime-only solver state.
 *
 * Exact solver output is deliberately kept outside the repository. Each run
 * replaces the previous checkpoint for that algorithm/density pair, and the
 * viewer retrieves it through the solver API instead of a packaged asset.
 */
export const LIVE_ROOT = join(tmpdir(), "ryugu-cauchy-gravity-live");

mkdirSync(LIVE_ROOT, { recursive: true });

export type LiveSolver = "werner" | "mascon" | "rtfp" | "carlson" | "carlsonalpha";

const RECORD_NAMES: Record<LiveSolver, Record<string, string>> = {
  werner: {
    uniform: "werner-uniform.bin",
  },
  mascon: {
    cauchy: "mascon-cauchy.bin",
    elliptic: "mascon-elliptic.bin",
  },
  rtfp: {
    cauchy: "rtfp-cauchy.bin",
    constant: "rtfp-uniform.bin",
  },
  carlson: {
    cauchy: "carlson-cauchy.bin",
    constant: "carlson-uniform.bin",
  },
  carlsonalpha: {
    elliptic: "carlsonalpha-elliptic.bin",
    constant: "carlsonalpha-uniform.bin",
  },
};

export function liveRecordPath(solver: LiveSolver, density: string): string {
  const name = RECORD_NAMES[solver]?.[density];
  if (!name) {
    throw new Error(`${solver} does not expose a ${density} record`);
  }
  return join(LIVE_ROOT, name);
}

export function liveWorkPath(
  solver: LiveSolver,
  density: string,
  kind: "order" | "log" | "checkpoint",
) {
  return join(LIVE_ROOT, `.${solver}-${density}-${kind}`);
}
