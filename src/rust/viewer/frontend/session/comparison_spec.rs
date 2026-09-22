// ---------------------------------------------------------------------------
// Reference tracks for the comparison panel
// ---------------------------------------------------------------------------

struct CompareRef {
    name: &'static str,
    kind: &'static str,
    algo: &'static str,
    density_mode: &'static str,
}

fn compare_ref(key: &str) -> Option<CompareRef> {
    let (name, kind, algo, density_mode) = match key {
        "werner" => ("Werner", "uniform density", "werner", "uniform"),
        "rtfp-cauchy" => ("RT-FP", "Cauchy density", "rtfp", "cauchy"),
        "rtfp-constant" => ("RT-FP", "uniform density", "rtfp", "constant"),
        "mascon-cauchy" => ("Mascon", "Cauchy density", "mascon", "cauchy"),
        "mascon-elliptic" => ("Mascon", "fractional Cauchy density", "mascon", "elliptic"),
        "carlson-cauchy" => ("Carlson", "Cauchy density", "carlson", "cauchy"),
        "carlson-constant" => ("Carlson", "uniform density", "carlson", "constant"),
        "carlsonalpha-elliptic" => (
            "CarlsonAlpha",
            "fractional Cauchy density",
            "carlsonalpha",
            "elliptic",
        ),
        _ => return None,
    };
    Some(CompareRef {
        name,
        kind,
        algo,
        density_mode,
    })
}

/// Werner and Mascon are the reference tracks: every other algorithm is
/// measured against them, so they carry no outgoing comparison of their own.
/// The RT-FP/Carlson pair is framed with Carlson as the baseline, so Carlson
/// does not list RT-FP while RT-FP lists Carlson first.
fn compare_order(algo: &str, mode: &str) -> &'static [&'static str] {
    match (algo, mode) {
        ("rtfp", "elliptic") => &["carlsonalpha-elliptic", "mascon-elliptic", "werner"],
        ("rtfp", "cauchy") => &["carlson-cauchy", "mascon-cauchy", "werner"],
        ("rtfp", "constant") => &["werner"],
        ("carlson", "cauchy") => &["mascon-cauchy", "werner"],
        ("carlson", "constant") => &["werner"],
        ("carlsonalpha", "elliptic") => &["mascon-elliptic", "werner"],
        ("carlsonalpha", "constant") => &["werner"],
        ("mascon", "elliptic") => &["werner"],
        ("mascon", "cauchy") => &["werner"],
        _ => &[],
    }
}

/// Every bake path shares the observation surface and the face indexing: ‖H‖_F
/// at each mesh vertex (`vertex + n_vertex · h`), averaged over the face's three
/// vertices. Face f is the same triangle in every record, so what is left is
/// solver difference, not sampling.
const SAMPLING_NOTE: &str = " · identical runtime sampling: face-centre ‖H‖_F with face normals";

fn same_standoff(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3
}
