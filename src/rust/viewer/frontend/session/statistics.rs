// ---------------------------------------------------------------------------
// Colour-window statistics and face-by-face differences
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
struct ScalarStats {
    min: f64,
    max: f64,
    cv: String,
    lo: f64,
    hi: f64,
    heavy: bool,
}

impl ScalarStats {
    /// Exactly the line the viewer appends to its status text.
    fn describe(&self) -> String {
        format!(
            " · ‖H‖_F min={} max={} std/mean={} · window [{}, {}] {}",
            fmt_number(self.min),
            fmt_number(self.max),
            self.cv,
            fmt_number(self.lo),
            fmt_number(self.hi),
            if self.heavy { "asinh" } else { "linear" },
        )
    }
}

/// `toPrecision(3)`-style rendering used by the status line.
fn fmt_number(value: f64) -> String {
    if !value.is_finite() {
        return "?".to_string();
    }
    if value == 0.0 {
        return "0.00".to_string();
    }
    let magnitude = value.abs();
    if (1e-2..1e3).contains(&magnitude) {
        let exponent = magnitude.log10().floor() as i32;
        let decimals = (2 - exponent).max(0) as usize;
        format!("{value:.decimals$}")
    } else {
        // `toExponential(2)`, with the JavaScript exponent spelling.
        let exponent = magnitude.log10().floor() as i32;
        let mantissa = value / 10f64.powi(exponent);
        format!(
            "{mantissa:.2}e{}{}",
            if exponent < 0 { "-" } else { "+" },
            exponent.abs()
        )
    }
}

fn quantile(sorted: &[f32], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last = sorted.len() - 1;
    let index = (((last as f64) * q).round() as usize).min(last);
    sorted[index] as f64
}

/// Colour-window statistics, so the panel explains what the renderer chose.
fn bake_scalar_stats(scalars: &[f32]) -> Option<ScalarStats> {
    let mut finite: Vec<f32> = scalars
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f32::total_cmp);
    let n = finite.len();
    let min = finite[0] as f64;
    let max = finite[n - 1] as f64;
    let sum: f64 = finite.iter().map(|value| *value as f64).sum();
    let mean = sum / n as f64;
    let variance: f64 = finite
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum();
    let std = (variance / n as f64).sqrt();
    let lo = quantile(&finite, 0.02);
    let med = quantile(&finite, 0.5);
    let hi = quantile(&finite, 0.98);
    // Same rule as the renderer (all algorithms share one colour window): asinh
    // only when the median face would sit in the bottom eighth of a linear ramp
    // — i.e. when the body would collapse onto one colour.
    let heavy = lo > 0.0 && hi > lo && (med - lo) / (hi - lo) < 0.125;
    Some(ScalarStats {
        min,
        max,
        cv: if mean > 0.0 {
            format!("{:.2}", std / mean)
        } else {
            "?".to_string()
        },
        lo,
        hi,
        heavy,
    })
}

#[derive(Clone, Debug, Default, PartialEq)]
struct DiffStats {
    compared: usize,
    total: usize,
    rel_eps: f64,
    over: usize,
    med: f64,
    max: f64,
}

/// Face-by-face difference between two records.
///
/// The tolerance has to be *relative*: ‖H‖_F is ~1e-6, so an absolute test could
/// never fire and the panel would report "0 differences" for records that
/// disagree by tens of percent per face.
fn scalar_diff_stats(a: &ParsedRecord, b: &ParsedRecord, rel_eps: f64) -> Option<DiffStats> {
    let n = a.total.min(b.total);
    if n == 0 {
        return None;
    }
    let mut relative: Vec<f32> = Vec::with_capacity(n);
    for index in 0..n {
        let (sa, sb) = (a.scalars[index], b.scalars[index]);
        if !sa.is_finite() || !sb.is_finite() {
            continue;
        }
        let sa = sa as f64;
        let sb = sb as f64;
        let denominator = sa.abs().max(sb.abs()).max(1e-300);
        relative.push(((sa - sb).abs() / denominator) as f32);
    }
    if relative.is_empty() {
        return None;
    }
    relative.sort_by(f32::total_cmp);
    let over = relative
        .iter()
        .filter(|value| **value as f64 > rel_eps)
        .count();
    Some(DiffStats {
        compared: relative.len(),
        total: n,
        rel_eps,
        over,
        med: quantile(&relative, 0.5),
        max: relative[relative.len() - 1] as f64,
    })
}

