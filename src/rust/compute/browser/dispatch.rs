// ---------------------------------------------------------------------------
// Dispatch geometry
// ---------------------------------------------------------------------------

fn pipeline_url(path: &str, base_url: &str) -> String {
    if path.starts_with("runtime:") {
        return path.to_string();
    }
    let mut url = String::with_capacity(path.len() + 48);
    if path.starts_with('/') {
        url.push_str(path);
    } else {
        url.push_str(base_url.trim_end_matches('/'));
        url.push('/');
        url.push_str(path.trim_start_matches("./"));
    }
    url
}

fn grid2d(count: usize, workgroup: usize) -> (u32, u32) {
    let groups = count.div_ceil(workgroup).max(1);
    let x = groups.min(65535);
    let y = groups.div_ceil(x);
    (x as u32, y as u32)
}

/// Dispatch grid for the `analytic` entry point, which is one *workgroup* per
/// observation point: it reduces 64 lanes of face work per point inside the
/// workgroup, so it must be sized in points, not in invocations. Sizing it like
/// the invocation-parallel passes (`÷ 64`) is not an error WebGPU reports — the
/// pipeline is valid and the dispatch succeeds — it just leaves all but the
/// first 1/64 of every block unwritten, i.e. zero.
fn point_grid(points: usize) -> (u32, u32) {
    grid2d(points, 1)
}

fn block_size(algorithm: &str) -> usize {
    match algorithm {
        "rtfp" | "carlson" | "carlsonalpha" => RAY_BLOCK_FACES,
        "mascon" => MASCON_BLOCK_FACES,
        _ => ANALYTIC_BLOCK_FACES,
    }
}

// ---------------------------------------------------------------------------
// Asset parsing
// ---------------------------------------------------------------------------

// `parseFaceAsset`: an RWR1/RYC1 record table of 64-byte face records.
