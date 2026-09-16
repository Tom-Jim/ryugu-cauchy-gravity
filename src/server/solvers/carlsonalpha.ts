/**
 * CarlsonAlpha endpoint: general positive Cauchy exponents.
 *
 * This is deliberately separate from the legacy Carlson jump-surface solver.
 * It uses the same deterministic ray geometry but evaluates the regularized
 * radial finite part directly, with the Carlson RF/RD/RJ/RC backend used for
 * the library verification path.
 */
import { join } from "node:path";
import { ROOT, createGpuSolver } from "./gpu_solver";
import { liveRecordPath, liveWorkPath } from "./live_paths";

export const carlsonalpha = createGpuSolver({
  id: "carlson-alpha",
  label: "CarlsonAlpha",
  defaultDensity: "elliptic",
  outByMode: {
    elliptic: liveRecordPath("carlsonalpha", "elliptic"),
    constant: liveRecordPath("carlsonalpha", "constant"),
  },
  densityByMode: {
    elliptic: join(ROOT, "assets/density/cauchy_elliptic.toml"),
    constant: join(ROOT, "assets/density/cauchy.toml"),
  },
  orderByMode: {
    elliptic: liveWorkPath("carlsonalpha", "elliptic", "order"),
    constant: liveWorkPath("carlsonalpha", "constant", "order"),
  },
  logByMode: {
    elliptic: liveWorkPath("carlsonalpha", "elliptic", "log"),
    constant: liveWorkPath("carlsonalpha", "constant", "log"),
  },
  checkpointByMode: {
    elliptic: liveWorkPath("carlsonalpha", "elliptic", "checkpoint"),
    constant: liveWorkPath("carlsonalpha", "constant", "checkpoint"),
  },
  // Mascon uses the TOML weights as written when total_mass_target is zero.
  // Keep CarlsonAlpha on the same scale instead of falling back to rho(0).
  extraArgs: ["--normalize", "raw"],
  modeArgs: {},
});

export const bootCarlsonAlphaStatus = () => carlsonalpha.boot();
export const carlsonalphaStatusResponse = () => carlsonalpha.statusResponse();
export const startCarlsonAlphaBake = (
  mode: "restart" | "resume",
  standoffMm?: number,
  density?: string,
) => carlsonalpha.start(mode, standoffMm, density);
