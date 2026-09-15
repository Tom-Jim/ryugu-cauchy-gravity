/**
 * RT-FP solver endpoint.
 *
 * RT-FP = "radial tensor finite part". It splits the gravity-gradient tensor of
 * an arbitrary density field into
 *
 *   H = rho(x) * W(x)  +  G * sum_q w_q T(u_q) * sum_k w_k R_k(u_q)
 *
 * i.e. an analytic polyhedral tensor for the uniform part plus a directional
 * quadrature carrying the density deviation. All of that runs in WGSL compute
 * shaders; the host process only marshals buffers.
 */
import { join } from "node:path";
import { ROOT, createGpuSolver } from "./gpu_solver";

const DIRECTIONS = Number(process.env.RTFP_DIRECTIONS ?? 288);

export const rtfp = createGpuSolver({
  id: "ray",
  label: "RT-FP",
  outByMode: {
    cauchy: join(ROOT, "assets/records/rtfp_faces.bin"),
    constant: join(ROOT, "assets/records/rtfp_constant_faces.bin"),
  },
  order: join(ROOT, "assets/records/.rtfp_order.bin"),
  log: join(ROOT, "assets/records/.rtfp_progress.log"),
  extraArgs: [],
  // The direction count only enters the Cauchy remainder quadrature; the
  // constant-density control case has no kernels, so it needs no directions.
  modeArgs: { cauchy: ["--directions", String(DIRECTIONS)] },
});

export const bootRtfpStatus = () => rtfp.boot();
export const rtfpStatusResponse = () => rtfp.statusResponse();
export const startRtfpBake = (
  mode: "restart" | "resume",
  standoffMm?: number,
  density?: string,
) => rtfp.start(mode, standoffMm, density);
