// ---------------------------------------------------------------------------
// Algorithm table
// ---------------------------------------------------------------------------

struct Algo {
    name: &'static str,
    /// Label used when the algorithm has no selectable density model.
    short: &'static str,
    /// Panel title used when the algorithm has no selectable density model.
    title: &'static str,
    /// Selectable density models; a single entry means "fixed".
    density_modes: &'static [&'static str],
    default_density: &'static str,
    next: &'static str,
    compare: bool,
}

static WERNER: Algo = Algo {
    name: "Werner",
    short: "Werner · uniform density",
    title: "Werner · uniform density · analytic polyhedral gradient",
    density_modes: &[],
    default_density: "constant",
    next: "Switch to Mascon (voxel)",
    compare: true,
};
static MASCON: Algo = Algo {
    name: "Mascon",
    short: "",
    title: "",
    density_modes: &["elliptic", "cauchy"],
    default_density: "elliptic",
    next: "Switch to RT-FP",
    compare: true,
};
static RTFP: Algo = Algo {
    name: "RT-FP",
    short: "",
    title: "",
    density_modes: &["cauchy", "constant"],
    default_density: "cauchy",
    next: "Switch to Carlson",
    compare: true,
};
static CARLSON: Algo = Algo {
    name: "Carlson",
    short: "",
    title: "",
    density_modes: &["cauchy", "constant"],
    default_density: "cauchy",
    next: "Switch to CarlsonAlpha",
    compare: true,
};
static CARLSON_ALPHA: Algo = Algo {
    name: "CarlsonAlpha",
    short: "",
    title: "",
    density_modes: &["elliptic", "constant"],
    default_density: "elliptic",
    next: "Switch to Werner",
    compare: true,
};

fn algo_spec(key: &str) -> &'static Algo {
    match key {
        "mascon" => &MASCON,
        "rtfp" => &RTFP,
        "carlson" => &CARLSON,
        "carlsonalpha" => &CARLSON_ALPHA,
        _ => &WERNER,
    }
}

fn density_label(mode: &str) -> &'static str {
    match mode {
        "uniform" | "constant" => "uniform",
        "elliptic" => "fractional Cauchy",
        _ => "Cauchy",
    }
}
