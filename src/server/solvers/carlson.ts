/**
 * Carlson solver endpoint — the independent verification algorithm.
 *
 * Carlson's formulation describes a piecewise-constant density field purely by
 * its jump surfaces, so the gravity-gradient tensor is one surface integral
 *
 *   H_ij(x) = G * sum_F d_rho_F * n_j(F) * I_F[i](x)
 *
 * over a star-cone tetrahedral decomposition of the body, with the same face
 * kernel the analytic tensor uses. There is no ray tracing, no directional
 * quadrature and no radial remainder term, which is exactly why it is useful as
 * a check on RT-FP: the two share no numerical machinery beyond the mesh.
 */
import { join } from "node:path";
import { ROOT, createGpuSolver } from "./gpu_solver";

export const carlson = createGpuSolver({
  id: "carlson",
  label: "Carlson",
  outByMode: {
    cauchy: join(ROOT, "assets/records/carlson_cauchy_faces.bin"),
    constant: join(ROOT, "assets/records/carlson_constant_faces.bin"),
  },
  order: join(ROOT, "assets/records/.carlson_order.bin"),
  log: join(ROOT, "assets/records/.carlson_progress.log"),
  extraArgs: [],
  modeArgs: {},
});

export const bootCarlsonStatus = () => carlson.boot();
export const carlsonStatusResponse = () => carlson.statusResponse();
export const startCarlsonBake = (
  mode: "restart" | "resume",
  standoffMm?: number,
  density?: string,
) => carlson.start(mode, standoffMm, density);
