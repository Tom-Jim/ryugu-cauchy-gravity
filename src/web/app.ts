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
        saveCurrent: () => session?.on_save_current(),
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
          const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
          style.textContent = ".diagnostic-grid{stroke:rgba(140,233,255,.13);stroke-width:1}.diagnostic-axis{stroke:rgba(231,247,255,.65);stroke-width:1}.diagnostic-tick{fill:#8da7b8;font:10px monospace}.diagnostic-label,.diagnostic-legend{fill:#e7f7ff;font:11px monospace}.diagnostic-path{fill:none;stroke-width:2.5;vector-effect:non-scaling-stroke}.diagnostic-point{stroke:#07101d;stroke-width:1.25}";
          copy.prepend(style);
          const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = diagnostics.downloadName;
          link.click();
          URL.revokeObjectURL(url);
        },
      };
    },
  });
  app.config.errorHandler = (error) => showFatalError(error);
  app.mount(root);
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
