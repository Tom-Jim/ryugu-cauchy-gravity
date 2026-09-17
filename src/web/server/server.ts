/**
 * The only server role is static distribution.
 *
 * All numerical work lives in the Rust/WASM pipelines loaded by the browser.
 * Keeping this entry point unconditional is important: an environment flag must
 * never accidentally bring back a server-side solver or a polling API.
 */
import { serveStatic } from "./static";

const PORT = Number(Bun.env.PORT ?? 3000);
Bun.serve({
  port: PORT,
  hostname: "0.0.0.0",
  fetch: serveStatic,
});

console.log(`http://127.0.0.1:${PORT} (static package host)`);
