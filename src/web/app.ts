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
};

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
  compute: { visible: false, text: "计算中 · 0.0%" },
  mobile: { visible: false },
  algo: {
    label: "Werner · uniform density",
    button: "Next algorithm",
    panelTitle: "Werner · uniform density · analytic polyhedral gradient",
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
  status: "Reading status…",
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
