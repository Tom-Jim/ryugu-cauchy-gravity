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

export const carlsonalpha = createGpuSolver({
  id: "carlson-alpha",
  label: "CarlsonAlpha",
  defaultDensity: "elliptic",
  outByMode: {
    elliptic: join(ROOT, "assets/records/carlsonalpha_elliptic_faces.bin"),
    constant: join(ROOT, "assets/records/carlsonalpha_constant_faces.bin"),
  },
  densityByMode: {
    elliptic: join(ROOT, "assets/density/cauchy_elliptic.toml"),
    constant: join(ROOT, "assets/density/cauchy.toml"),
  },
  order: join(ROOT, "assets/records/.carlsonalpha_order.bin"),
  log: join(ROOT, "assets/records/.carlsonalpha_progress.log"),
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
