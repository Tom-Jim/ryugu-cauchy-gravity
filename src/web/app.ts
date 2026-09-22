type SessionController = {
  on_algo(): void;
  on_bake(): void;
  on_density(): void;
  on_reload(): void;
  on_standoff_input(value: number): void;
  on_standoff_commit(): void;
  on_mobile_dismiss(): void;
  on_save_current(): void;
  on_download_current(): void;
  on_delete_current(): void;
  on_delete_item(id: string): void;
  run_diagnostic(kind: string): void;
  evaluate_diagnostic(options: Record<string, unknown>): Promise<Uint8Array>;
  set_asset_selection(model: Uint8Array, modelPath: string, cauchyText: string, ellipticText: string): void;
  set_model_scale(scaleMetersPerUnit: number): void;
  set_rotation_quaternion(x: number, y: number, z: number, w: number): void;
  set_rotation_period(periodHours: number): void;
};

type DiagnosticPoint = { x: number; y: number };
type DiagnosticSeries = { name: string; color: string; facet: string; dash: string; points: DiagnosticPoint[] };
type DiagnosticScale = "linear" | "log";

// View layer only.
//
// `viewerUi` is the render state and `saved` is the persisted-record list. Both
// are written by the Rust session controller (`src/rust/viewer/frontend/session/`), which
// reads and writes the IndexedDB rows itself (`src/rust/viewer/frontend/store.rs`); this file
// only declares the reactive objects, groups the saved rows for display, and
// forwards button events to the controller. No value here is computed: the
// labels the panel shows are rendered by Rust so there is one implementation of
// every number and every string.

import { computed, createApp, reactive } from "vue";

let session: SessionController | null = null;
let mounted = false;

function errorText(error: unknown): string {
  if (error instanceof Error) return error.stack || error.message;
  return String(error);
}

export const uiState = reactive({
  showParams: false,
  showSaved: false,
});

export const viewerUi = reactive({
  message: { visible: false, text: "" },
  compute: { visible: false, text: "Computing - 0.0%" },
  mobile: { visible: false },
  algo: {
    label: "Werner - uniform density",
    button: "Next algorithm",
    panelTitle: "Werner - uniform density - analytic polyhedral gradient",
  },
  standoff: {
    value: "16.00 m",
    pos: 0,
    staticHidden: true,
    staticText: "Rust WASM / WebGPU",
    staticTitle: "",
    title: "Observation height",
    record: "",
  },
  buttons: {
    bake: { hidden: false, disabled: false, text: "Recompute" },
    density: { hidden: true, disabled: false, text: "Switch to uniform density" },
    reload: { hidden: true, disabled: false, text: "Reload colours" },
  },
  barPercent: 0,
  status: "Reading status...",
  compare: { visible: false, text: "" },
});

export const saved = reactive({
  items: [] as Array<Record<string, any>>,
  loading: false,
  busy: false,
  message: "",
  canSave: false,
  hasCurrentSaved: false,
  current: {
    algo: "werner",
    algorithm: "Werner",
    density: "uniform",
    standoffMm: 16000,
    done: false,
  },
});

export const diagnostics = reactive({
  visible: false,
  kind: "",
  title: "",
  xLabel: "",
  yLabel: "",
  xScale: "log" as DiagnosticScale,
  yScale: "log" as DiagnosticScale,
  busy: false,
  ready: false,
  progress: 0,
  status: "",
  error: "",
  summary: "",
  downloadName: "diagnostic.svg",
  series: [] as DiagnosticSeries[],
});

type AssetItem = { id: string; kind: "model" | "density"; name: string; path?: string; bytes?: Uint8Array; selected: boolean; };
export const assets = reactive({
  visible: false, loading: false, busy: false, dragging: false, message: "",
  currentModel: "ryugu.glb",
  models: [] as AssetItem[], densities: [] as AssetItem[],
});

// Ryugu reference values (Hayabusa2 shape/mass scale). These are display and
// session parameters; changing them never mutates the Bevy or WGSL pipelines.
export const modelParameters = reactive({
  massKg: 4.50e11,
  meanDensityKgM3: 1190,
  scaleMetersPerUnit: 1.0,
  rotationPeriodHours: 7.63,
  rotationQuaternionX: 0.0,
  rotationQuaternionY: 0.0,
  rotationQuaternionZ: 0.0,
  rotationQuaternionW: 1.0,
});
const MODEL_DEFAULTS: Record<string, Partial<typeof modelParameters>> = {
  "ryugu.glb": { massKg: 4.50e11, meanDensityKgM3: 1190, scaleMetersPerUnit: 1, rotationPeriodHours: 7.63, rotationQuaternionX: 0, rotationQuaternionY: 0, rotationQuaternionZ: 0, rotationQuaternionW: 1 },
  "Phobos.glb": { massKg: 1.0659e16, meanDensityKgM3: 1876, scaleMetersPerUnit: 1, rotationPeriodHours: 7.653, rotationQuaternionX: 0, rotationQuaternionY: 0, rotationQuaternionZ: 0, rotationQuaternionW: 1 },
  "Deimos.glb": { massKg: 1.4762e15, meanDensityKgM3: 1471, scaleMetersPerUnit: 1, rotationPeriodHours: 30.298, rotationQuaternionX: 0, rotationQuaternionY: 0, rotationQuaternionZ: 0, rotationQuaternionW: 1 },
};
function storageKeyForModel(modelName: string): string {
  return `ryugu-model-params-v3:${modelName.toLowerCase()}`;
}
function applyModelDefaults(name: string) {
  const defaults = MODEL_DEFAULTS[name] ?? MODEL_DEFAULTS["ryugu.glb"]!;
  for (const [key, value] of Object.entries(defaults)) if (value !== undefined) (modelParameters as any)[key] = value;
  loadModelParameters(name);
  saveModelParameters(name);
}
function loadModelParameters(modelName: string = "ryugu.glb") {
  try {
    localStorage.removeItem("ryugu-model-parameters-v2");
    const saved = JSON.parse(localStorage.getItem(storageKeyForModel(modelName)) || "null");
    if (saved && typeof saved === "object") {
      for (const key of Object.keys(modelParameters) as Array<keyof typeof modelParameters>) {
        const value = Number(saved[key]);
        if (Number.isFinite(value)) {
          if (key.startsWith("rotationQuaternion")) {
            modelParameters[key] = value;
          } else if (value > 0) {
            modelParameters[key] = value;
          }
        }
      }
    }
  } catch { /* use reference defaults */ }
}
function saveModelParameters(modelName?: string) {
  const name = modelName ?? assets?.currentModel ?? "ryugu.glb";
  try {
    localStorage.setItem(storageKeyForModel(name), JSON.stringify(modelParameters));
  } catch { /* ignore storage errors */ }
}
loadModelParameters();
function normalizeAndSyncQuaternion() {
  let x = modelParameters.rotationQuaternionX;
  let y = modelParameters.rotationQuaternionY;
  let z = modelParameters.rotationQuaternionZ;
  let w = modelParameters.rotationQuaternionW;
  const norm = Math.hypot(x, y, z, w);
  if (norm > 1e-9) {
    x = Number((x / norm).toFixed(4));
    y = Number((y / norm).toFixed(4));
    z = Number((z / norm).toFixed(4));
    const rem = Math.max(0, 1 - (x * x + y * y + z * z));
    w = Number((Math.sign(w || 1) * Math.sqrt(rem)).toFixed(4));
  } else {
    x = 0; y = 0; z = 0; w = 1;
  }
  modelParameters.rotationQuaternionX = x;
  modelParameters.rotationQuaternionY = y;
  modelParameters.rotationQuaternionZ = z;
  modelParameters.rotationQuaternionW = w;
  saveModelParameters();
  session?.set_rotation_quaternion(x, y, z, w);
}
function applyPhysicalParameters(text: string): string {
  let updated = text
    .replace(/(^|\n)total_mass_target\s*=\s*[-+0-9.eE]+/m, `$1total_mass_target = ${modelParameters.massKg}`)
    .replace(/(^|\n)bulk_density_ref\s*=\s*[-+0-9.eE]+/m, `$1bulk_density_ref = ${modelParameters.meanDensityKgM3}`)
    .replace(/(^|\n)mean_density\s*=\s*[-+0-9.eE]+/m, `$1mean_density = ${modelParameters.meanDensityKgM3}`);
  if (!/(^|\n)mean_density\s*=/.test(updated)) {
    updated += `\nmean_density = ${modelParameters.meanDensityKgM3}\n`;
  }
  return updated;
}
async function syncPhysicalParametersToSession(): Promise<void> {
  const selectedModels = assets.models.filter((item) => item.selected);
  const model = selectedModels[0];
  const densities = assets.densities.filter((item) => item.selected);
  const elliptic = densities.find((item) => item.name.toLowerCase().includes("elliptic")) ?? assets.densities[0];
  if (elliptic && session) {
    try {
      const ellipticBytes = await bytesFor(elliptic);
      const tomlText = applyPhysicalParameters(new TextDecoder().decode(ellipticBytes));
      session.set_asset_selection(
        model ? await bytesFor(model) : new Uint8Array(),
        model?.path?.startsWith("assets/models/") ? model.path.slice("assets/".length) : "",
        tomlText,
        tomlText,
      );
    } catch { /* ignore if not ready */ }
  }
}

const ASSET_DB = "ryugu-cauchy-gravity";
const ASSET_STORE = "resources";
const ASSET_LOCK = "ryugu-cauchy-gravity:asset-session";
const ASSET_LOCK_TTL_MS = 8_000;
const ASSET_LOCK_HEARTBEAT_MS = 2_000;
const assetSessionToken = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
let assetDb: IDBDatabase | null = null;
let assetSessionOwned = false;
let assetSessionReady: Promise<boolean>;
let assetHeartbeat: number | undefined;
let assetSessionError = "";
const assetDefaults: AssetItem[] = [
  { id: "model:ryugu.glb", kind: "model", name: "ryugu.glb", path: "assets/models/ryugu.glb", selected: true },
  { id: "model:Deimos.glb", kind: "model", name: "Deimos.glb", path: "assets/models/Deimos.glb", selected: false },
  { id: "model:Phobos.glb", kind: "model", name: "Phobos.glb", path: "assets/models/Phobos.glb", selected: false },
  { id: "density:cauchy_elliptic.toml", kind: "density", name: "cauchy_elliptic.toml", path: "assets/density/cauchy_elliptic.toml", selected: true },
];
function assetRequest(request: IDBRequest): Promise<any> {
  return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
}
function readAssetLock(): { token: string; heartbeat: number } | null {
  try {
    const value = JSON.parse(localStorage.getItem(ASSET_LOCK) || "null");
    return value && typeof value.token === "string" && Number.isFinite(value.heartbeat) ? value : null;
  } catch { return null; }
}
function writeAssetLock() {
  localStorage.setItem(ASSET_LOCK, JSON.stringify({ token: assetSessionToken, heartbeat: Date.now() }));
}
async function deleteAssetDatabase(): Promise<void> {
  if (assetDb) { assetDb.close(); assetDb = null; }
  await new Promise<void>((resolve) => {
    const request = indexedDB.deleteDatabase(ASSET_DB);
    request.onsuccess = request.onerror = request.onblocked = () => resolve();
  });
}
async function startAssetSession(): Promise<boolean> {
  const current = readAssetLock();
  if (current && current.token !== assetSessionToken && Date.now() - current.heartbeat < ASSET_LOCK_TTL_MS) return false;
  try {
    writeAssetLock();
    // Re-check after the write so two tabs opening at the same time cannot both win.
    const confirmed = readAssetLock();
    if (!confirmed || confirmed.token !== assetSessionToken) return false;
    await deleteAssetDatabase();
    assetSessionOwned = true;
    assetHeartbeat = window.setInterval(() => {
      if (assetSessionOwned) writeAssetLock();
    }, ASSET_LOCK_HEARTBEAT_MS);
    return true;
  } catch (error) {
    assetSessionError = `IndexedDB 初始化失败：${error instanceof Error ? error.message : String(error)}`;
    return false;
  }
}
function stopAssetSession() {
  if (!assetSessionOwned) return;
  assetSessionOwned = false;
  if (assetHeartbeat !== undefined) window.clearInterval(assetHeartbeat);
  assetHeartbeat = undefined;
  if (readAssetLock()?.token === assetSessionToken) localStorage.removeItem(ASSET_LOCK);
  // pagehide cannot await this operation; starting it still lets the browser
  // remove the session database before the next tab opens.
  void deleteAssetDatabase();
}
assetSessionReady = startAssetSession();
window.addEventListener("pagehide", stopAssetSession, { once: true });
window.addEventListener("beforeunload", stopAssetSession, { once: true });
async function openAssetDb(): Promise<IDBDatabase> {
  if (!assetSessionOwned && !(await assetSessionReady)) throw new Error("另一个标签页正在使用资源库，请关闭它后重试。");
  if (assetDb) return assetDb;
  const request = indexedDB.open(ASSET_DB, 3);
  request.onupgradeneeded = () => {
    if (!request.result.objectStoreNames.contains(ASSET_STORE)) request.result.createObjectStore(ASSET_STORE, { keyPath: "id" });
  };
  assetDb = await assetRequest(request);
  return assetDb!;
}
async function readAssetRecords(): Promise<AssetItem[]> {
  const db = await openAssetDb();
  return (await assetRequest(db.transaction(ASSET_STORE).objectStore(ASSET_STORE).getAll())) as AssetItem[];
}
async function putAsset(item: AssetItem): Promise<void> {
  const db = await openAssetDb();
  const record = { id: item.id, kind: item.kind, name: item.name, path: item.path, bytes: item.bytes, selected: item.selected };
  await assetRequest(db.transaction(ASSET_STORE, "readwrite").objectStore(ASSET_STORE).put(record));
}
async function deleteAsset(id: string): Promise<void> {
  const db = await openAssetDb();
  await assetRequest(db.transaction(ASSET_STORE, "readwrite").objectStore(ASSET_STORE).delete(id));
}
async function ensureAssetLibrary(): Promise<void> {
  assets.loading = true;
  assets.message = "";
  try {
    if (!(await assetSessionReady)) throw new Error(assetSessionError || "另一个标签页正在使用资源库，请关闭它后刷新页面。");
    const existing = await readAssetRecords();
    const byId = new Map(existing.map((item) => [item.id, item]));
    for (const item of assetDefaults) if (!byId.has(item.id)) await putAsset(item);
    const all = await readAssetRecords();
    for (const item of all.filter((candidate) => candidate.kind === "density" && candidate.id.startsWith("density:"))) {
      item.selected = true;
      await putAsset(item);
    }
    const finalRecords = await readAssetRecords();
    const storedModels = finalRecords.filter((item) => item.kind === "model");
    if (storedModels.filter((item) => item.selected).length !== 1) {
      const fallback = storedModels.find((item) => item.id === "model:ryugu.glb") ?? storedModels[0];
      for (const item of storedModels) {
        item.selected = item.id === fallback?.id;
        await putAsset(item);
      }
    }
    const normalizedRecords = await readAssetRecords();
    assets.models = normalizedRecords.filter((item) => item.kind === "model");
    assets.densities = normalizedRecords.filter((item) => item.kind === "density");
    assets.currentModel = assets.models.find((item) => item.selected)?.name ?? "ryugu.glb";
    // The coefficient is a model-to-Bevy unit conversion. Never carry a
    // satellite's conversion (for example Deimos=15) into Ryugu's default
    // asset, whose shipped GLB is already in metre-scale units.
    if (assets.currentModel.toLowerCase() === "ryugu.glb") {
      modelParameters.scaleMetersPerUnit = 1.0;
      saveModelParameters();
    }
    assets.message = "";
  } catch (error) {
    assets.message = `IndexedDB 失败：${error instanceof Error ? error.message : String(error)}`;
  }
  finally { assets.loading = false; }
}
async function bytesFor(item: AssetItem): Promise<Uint8Array> {
  if (item.bytes) return item.bytes;
  if (!item.path) throw new Error(`${item.name} has no source data`);
  const response = await fetch(item.path, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Could not load ${item.path} (${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  item.bytes = bytes;
  await putAsset(item);
  return bytes;
}
async function applyAssetSelection(): Promise<void> {
  const selectedModels = assets.models.filter((item) => item.selected);
  const model = selectedModels[0];
  const densities = assets.densities.filter((item) => item.selected);
  const elliptic = densities.find((item) => item.name.toLowerCase().includes("elliptic"));
  if (selectedModels.length !== 1 || !model || densities.length !== 1 || !elliptic) {
    assets.message = "Select exactly one GLB and cauchy_elliptic.toml before using them.";
    return;
  }
  assets.busy = true;
  try {
    const modelBytes = await bytesFor(model);
    const ellipticBytes = await bytesFor(elliptic);
    session?.set_asset_selection(
      modelBytes,
      model.path?.startsWith("assets/models/") ? model.path.slice("assets/".length) : "",
      applyPhysicalParameters(new TextDecoder().decode(ellipticBytes)),
      applyPhysicalParameters(new TextDecoder().decode(ellipticBytes)),
    );
    assets.currentModel = model.name;
    assets.visible = false;
    assets.message = "";
    session?.set_model_scale(modelParameters.scaleMetersPerUnit);
    session?.set_rotation_quaternion(
      modelParameters.rotationQuaternionX,
      modelParameters.rotationQuaternionY,
      modelParameters.rotationQuaternionZ,
      modelParameters.rotationQuaternionW,
    );
    session?.set_rotation_period(modelParameters.rotationPeriodHours);
    session?.on_bake();
  } catch (error) { assets.message = String(error); }
  finally { assets.busy = false; }
}
function canApplyAssets(): boolean {
  const models = assets.models.filter((item) => item.selected);
  const densities = assets.densities.filter((item) => item.selected);
  return models.length === 1
    && densities.length === 1
    && densities.some((item) => item.name.toLowerCase().includes("elliptic"));
}
async function importFiles(fileList: FileList | File[]): Promise<void> {
  try {
    for (const file of Array.from(fileList)) {
      const lower = file.name.toLowerCase();
      const kind = lower.endsWith(".glb") ? "model" : lower.endsWith(".toml") ? "density" : null;
      if (!kind) continue;
      const id = `${kind}:${file.name}:${file.size}:${file.lastModified}`;
      await putAsset({ id, kind, name: file.name, bytes: new Uint8Array(await file.arrayBuffer()), selected: false });
    }
    await ensureAssetLibrary();
    if (!assets.message) assets.message = "Imported files are ready to select.";
  } catch (error) {
    assets.message = `IndexedDB 导入失败：${error instanceof Error ? error.message : String(error)}`;
  }
}

const savedGroups = computed(() => {
  const groups = new Map<string, { algorithm: string; items: Array<Record<string, any>> }>();
  for (const item of saved.items as Array<Record<string, any>>) {
    const group: { algorithm: string; items: Array<Record<string, any>> } =
      groups.get(item.algorithm) ?? { algorithm: item.algorithm, items: [] };
    group.items.push(item);
    groups.set(item.algorithm, group);
  }
  return [...groups.values()];
});

const diagnosticChart = computed(() => {
  const faceted = diagnostics.kind === "stability";
  const facetNames = faceted ? ["Uniform", "Cauchy", "Fractional Cauchy"] : ["all"];
  const panelWidth = faceted ? 370 : 960;
  const width = panelWidth * facetNames.length;
  const height = faceted ? 560 : 540;
  const pad = { left: faceted ? 56 : 76, right: 24, top: 68, bottom: 62 };
  const formatDecimal = (value: number) => {
    if (!Number.isFinite(value)) return "";
    const magnitude = Math.abs(value);
    if (magnitude >= 1000) return value.toFixed(0);
    if (magnitude >= 1) return value.toFixed(3);
    if (magnitude >= 0.01) return value.toFixed(5);
    if (magnitude === 0) return "0.00000";
    return value.toFixed(Math.min(12, Math.max(5, Math.ceil(-Math.log10(magnitude)) + 3)));
  };
  const toDomainValue = (value: number, scale: DiagnosticScale): number => (
    scale === "log" ? Math.log10(value) : value
  );
  const isPlottable = (value: number, scale: DiagnosticScale): boolean => (
    Number.isFinite(value) && (scale !== "log" || value > 0)
  );
  const expandDomain = (values: number[], scale: DiagnosticScale): [number, number] => {
    const valid = values.filter((value) => isPlottable(value, scale)).map((value) => toDomainValue(value, scale));
    if (valid.length === 0) return [0, 1];
    let low = Math.min(...valid);
    let high = Math.max(...valid);
    const span = high - low;
    if (span === 0) {
      const margin = scale === "log" ? 0.5 : Math.max(Math.abs(low) * 0.1, 1);
      low -= margin;
      high += margin;
    } else {
      const margin = Math.max(span * 0.08, scale === "log" ? 0.12 : 1e-9);
      low -= margin;
      high += margin;
    }
    if (scale === "linear") low = Math.max(0, low);
    return [low, high];
  };
  const tickValues = (low: number, high: number, scale: DiagnosticScale) => {
    if (scale !== "log") {
      return Array.from({ length: 5 }, (_, index) => low + (high - low) * index / 4);
    }
    const firstExponent = Math.ceil(low);
    const lastExponent = Math.floor(high);
    const exponents = Array.from(
      { length: Math.max(0, lastExponent - firstExponent + 1) },
      (_, index) => firstExponent + index,
    );
    if (exponents.length === 0) return [10 ** low, 10 ** high];
    if (exponents.length <= 9) return exponents.map((exponent) => 10 ** exponent);
    const step = Math.ceil(exponents.length / 8);
    return exponents.filter((_, index) => index % step === 0).map((exponent) => 10 ** exponent);
  };
  const formatTick = (value: number, scale: DiagnosticScale): string => {
    if (scale === "log") {
      const exponent = Math.floor(Math.log10(value));
      const mantissa = value / 10 ** exponent;
      return `${mantissa.toPrecision(mantissa >= 10 ? 2 : 1)}e${exponent}`;
    }
    return formatDecimal(value);
  };
  const panels = facetNames.map((facet, facetIndex) => {
    const selected = diagnostics.series.filter((series) => facet === "all" || series.facet === facet);
    const validPoint = (point: DiagnosticPoint) => (
      isPlottable(point.x, diagnostics.xScale) && isPlottable(point.y, diagnostics.yScale)
    );
    const points = selected.flatMap((series) => series.points).filter(validPoint);
    const innerWidth = panelWidth - pad.left - pad.right;
    const innerHeight = height - pad.top - pad.bottom;
    if (points.length === 0) {
      return { facet, offsetX: facetIndex * panelWidth, width: panelWidth, height, inner: { ...pad, innerWidth, innerHeight }, ticksX: [], ticksY: [], paths: [] };
    }
    let [minX, maxX] = expandDomain(points.map((point) => point.x), diagnostics.xScale);
    let [minY, maxY] = expandDomain(points.map((point) => point.y), diagnostics.yScale);
    const x = (value: number) => facetIndex * panelWidth + pad.left + ((toDomainValue(value, diagnostics.xScale) - minX) / (maxX - minX)) * innerWidth;
    const y = (value: number) => pad.top + innerHeight - ((toDomainValue(value, diagnostics.yScale) - minY) / (maxY - minY)) * innerHeight;
    const ticks = (low: number, high: number, scale: DiagnosticScale) => tickValues(low, high, scale)
      .filter((value) => {
        const transformed = toDomainValue(value, scale);
        return transformed >= low - 1e-9 && transformed <= high + 1e-9;
      })
      .map((value) => ({ value, label: formatTick(value, scale) }));
    const pathFor = (series: DiagnosticSeries) => {
      let path = "";
      let open = false;
      for (const point of series.points) {
        if (!validPoint(point)) {
          open = false;
          continue;
        }
        path += `${open ? "L" : "M"}${x(point.x).toFixed(2)},${y(point.y).toFixed(2)} `;
        open = true;
      }
      return path.trim();
    };
    return {
      facet,
      offsetX: facetIndex * panelWidth,
      width: panelWidth,
      height,
      inner: { ...pad, innerWidth, innerHeight },
      ticksX: ticks(minX, maxX, diagnostics.xScale).map((tick) => ({ ...tick, position: x(tick.value) })),
      ticksY: ticks(minY, maxY, diagnostics.yScale).map((tick) => ({ ...tick, position: y(tick.value) })),
      paths: selected.map((series) => ({
        ...series,
        path: pathFor(series),
        plottedPoints: series.points.filter(validPoint).map((point) => ({
          x: x(point.x),
          y: y(point.y),
        })),
      })),
    };
  });
  return { width, height, panels };
});

export function configureSession(controller: SessionController) {
  session = controller;
  session.set_model_scale(modelParameters.scaleMetersPerUnit);
  session.set_rotation_quaternion(
    modelParameters.rotationQuaternionX,
    modelParameters.rotationQuaternionY,
    modelParameters.rotationQuaternionZ,
    modelParameters.rotationQuaternionW,
  );
  session.set_rotation_period(modelParameters.rotationPeriodHours);
}

export function mountViewer() {
  if (mounted) return;
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) throw new Error("Viewer root #app is missing");
  const app = createApp({
    setup() {
      return {
        ui: viewerUi,
        saved,
        groups: savedGroups,
        onAlgo: () => session?.on_algo(),
        onBake: () => session?.on_bake(),
        onDensity: () => session?.on_density(),
        onReload: () => session?.on_reload(),
        onStandoffInput: (event: Event) => (
          session?.on_standoff_input(Number((event.target as HTMLInputElement | null)?.value) || 0)
        ),
        onStandoffCommit: () => session?.on_standoff_commit(),
        onMobileDismiss: () => session?.on_mobile_dismiss(),
        assets,
        modelParameters,
        updateModelParameter: (key: keyof typeof modelParameters, event: Event) => {
          const value = Number((event.target as HTMLInputElement | null)?.value);
          const isQuat = key.startsWith("rotationQuaternion");
          if (Number.isFinite(value) && (isQuat || value > 0)) {
            modelParameters[key] = value;
            if (isQuat) {
              normalizeAndSyncQuaternion();
            } else {
              saveModelParameters();
              if (key === "scaleMetersPerUnit") {
                session?.set_model_scale(value);
              } else if (key === "rotationPeriodHours") {
                session?.set_rotation_period(value);
              } else if (key === "meanDensityKgM3" || key === "massKg") {
                void syncPhysicalParametersToSession();
                session?.on_bake();
              }
            }
          }
        },
        resetQuaternion: () => {
          modelParameters.rotationQuaternionX = 0.0;
          modelParameters.rotationQuaternionY = 0.0;
          modelParameters.rotationQuaternionZ = 0.0;
          modelParameters.rotationQuaternionW = 1.0;
          saveModelParameters();
          session?.set_rotation_quaternion(0.0, 0.0, 0.0, 1.0);
        },
        openAssets: async () => { assets.visible = true; await ensureAssetLibrary(); },
        closeAssets: () => { assets.visible = false; },
        applyAssets: applyAssetSelection,
        canApplyAssets,
        openAssetPicker: () => document.querySelector<HTMLInputElement>("#asset-file-picker")?.click(),
        selectAsset: (item: AssetItem, kind: "model" | "density") => {
          const list = kind === "model" ? assets.models : assets.densities;
          if (kind === "model") {
            item.selected = true;
            assets.currentModel = item.name;
            applyModelDefaults(item.name);
            if (item.name.toLowerCase() === "ryugu.glb") {
              modelParameters.scaleMetersPerUnit = 1.0;
              saveModelParameters();
              session?.set_model_scale(1.0);
            }
            session?.set_rotation_quaternion(
              modelParameters.rotationQuaternionX,
              modelParameters.rotationQuaternionY,
              modelParameters.rotationQuaternionZ,
              modelParameters.rotationQuaternionW,
            );
            session?.set_rotation_period(modelParameters.rotationPeriodHours);
            for (const candidate of list) candidate.selected = candidate.id === item.id;
          } else {
            item.selected = !item.selected;
          }
          void putAsset(item);
        },
        removeAsset: async (item: AssetItem) => {
          if (item.id.startsWith("model:ryugu.glb") || item.id.startsWith("density:cauchy_elliptic.toml")) return;
          await deleteAsset(item.id);
          await ensureAssetLibrary();
        },
        importAssets: (event: Event) => { const files = (event.target as HTMLInputElement).files; if (files) void importFiles(files); },
        dropAssets: (event: DragEvent) => { event.preventDefault(); assets.dragging = false; if (event.dataTransfer?.files) void importFiles(event.dataTransfer.files); },
        dragAssets: (event: DragEvent) => { event.preventDefault(); assets.dragging = true; },
        leaveAssets: () => { assets.dragging = false; },
        uiState,
        toggleParams: () => { uiState.showParams = !uiState.showParams; },
        toggleSaved: () => { uiState.showSaved = !uiState.showSaved; },
        closeSaved: () => { uiState.showSaved = false; },
        saveCurrent: () => {
          uiState.showSaved = true;
          session?.on_save_current();
        },
        downloadCurrent: () => session?.on_download_current(),
        deleteCurrent: () => session?.on_delete_current(),
        deleteItem: (item: Record<string, any>) => session?.on_delete_item(String(item?.id ?? "")),
        diagnostics,
        diagnosticChart,
        openDiagnostic: (kind: string) => {
          diagnostics.visible = true;
          diagnostics.kind = kind;
          diagnostics.busy = true;
          diagnostics.ready = false;
          diagnostics.error = "";
          diagnostics.series = [];
          diagnostics.status = "Preparing diagnostic...";
          session?.run_diagnostic(kind);
        },
        closeDiagnostic: () => {
          diagnostics.visible = false;
        },
        downloadDiagnostic: () => {
          if (!diagnostics.ready) return;
          const svg = document.querySelector<SVGSVGElement>("#diagnostic-chart");
          if (!svg) return;
          const copy = svg.cloneNode(true) as SVGSVGElement;
          copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
          copy.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
          const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
          style.textContent = ".diagnostic-grid{stroke:rgba(140,233,255,.13);stroke-width:1}.diagnostic-axis{stroke:rgba(231,247,255,.65);stroke-width:1}.diagnostic-tick{fill:#8da7b8;font:10px monospace}.diagnostic-label,.diagnostic-legend{fill:#e7f7ff;font:11px monospace}.diagnostic-path{fill:none;stroke-width:2.5;vector-effect:non-scaling-stroke}.diagnostic-point{stroke:#07101d;stroke-width:1.25}";
          copy.prepend(style);
          const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = diagnostics.downloadName;
          link.click();
          window.setTimeout(() => URL.revokeObjectURL(url), 0);
        },
      };
    },
  });
  app.config.errorHandler = (error) => showFatalError(error);
  app.mount(root);
  loadModelParameters();
  root.dataset.mounted = "true";
  mounted = true;
}

export function showFatalError(error: unknown) {
  const text = errorText(error);
  viewerUi.compute.visible = false;
  viewerUi.buttons.bake.disabled = true;
  viewerUi.buttons.bake.text = "Unavailable";
  viewerUi.message.visible = true;
  viewerUi.message.text = text;

  // This remains visible even when Vue itself failed before it could remove
  // v-cloak or render the reactive error state.
  const fallback = document.querySelector<HTMLElement>("#boot-error");
  if (fallback) {
    fallback.textContent = text;
    fallback.hidden = false;
  }
}
