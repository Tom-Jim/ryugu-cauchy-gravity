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
  xScale: "linear" as const,
  yScale: "linear" as const,
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
  const expandDomain = (low: number, high: number): [number, number] => {
    const span = high - low;
    const margin = span === 0 ? Math.max(Math.abs(low) * 0.1, 1) : Math.max(Math.abs(span) * 0.08, 1e-9);
    return [low - margin, high + margin];
  };
  const panels = facetNames.map((facet, facetIndex) => {
    const selected = diagnostics.series.filter((series) => facet === "all" || series.facet === facet);
    const points = selected.flatMap((series) => series.points).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    const innerWidth = panelWidth - pad.left - pad.right;
    const innerHeight = height - pad.top - pad.bottom;
    if (points.length === 0) {
      return { facet, offsetX: facetIndex * panelWidth, width: panelWidth, height, inner: { ...pad, innerWidth, innerHeight }, ticksX: [], ticksY: [], paths: [] };
    }
    let [minX, maxX] = expandDomain(Math.min(...points.map((point) => point.x)), Math.max(...points.map((point) => point.x)));
    let [minY, maxY] = expandDomain(Math.min(...points.map((point) => point.y)), Math.max(...points.map((point) => point.y)));
    const x = (value: number) => facetIndex * panelWidth + pad.left + ((value - minX) / (maxX - minX)) * innerWidth;
    const y = (value: number) => pad.top + innerHeight - ((value - minY) / (maxY - minY)) * innerHeight;
    const ticks = (low: number, high: number) => Array.from({ length: 5 }, (_, index) => {
      const value = low + (high - low) * index / 4;
      return { value, label: formatDecimal(value) };
    });
    return {
      facet,
      offsetX: facetIndex * panelWidth,
      width: panelWidth,
      height,
      inner: { ...pad, innerWidth, innerHeight },
      ticksX: ticks(minX, maxX).map((tick) => ({ ...tick, position: x(tick.value) })),
      ticksY: ticks(minY, maxY).map((tick) => ({ ...tick, position: y(tick.value) })),
      paths: selected.map((series) => ({
        ...series,
        path: series.points
          .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
          .map((point, index) => `${index === 0 ? "M" : "L"}${x(point.x).toFixed(2)},${y(point.y).toFixed(2)}`)
          .join(" "),
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
          style.textContent = ".diagnostic-grid{stroke:rgba(140,233,255,.13);stroke-width:1}.diagnostic-axis{stroke:rgba(231,247,255,.65);stroke-width:1}.diagnostic-tick{fill:#8da7b8;font:10px monospace}.diagnostic-label,.diagnostic-legend{fill:#e7f7ff;font:11px monospace}.diagnostic-path{fill:none;stroke-width:2.5;vector-effect:non-scaling-stroke}";
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
