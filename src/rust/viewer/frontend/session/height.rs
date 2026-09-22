// ---------------------------------------------------------------------------
// Observation-surface height
//
// The viewer needs both near-surface samples and metre-scale overview samples.
// Mascon is a direct sum over roughly 5.2 m voxels, while the analytic and
// finite-part paths resolve much smaller standoffs. A single linear track
// would hide the low-height region, so every algorithm uses the same
// two-segment vertical slider:
//
//   lower half (track 0…500)    →     1 mm …   500 mm
//   upper half (track 500…1000) →   1 m    …    32 m
//
// The slider position is stored, never millimetres; `standoff_mm` remains the
// single source of truth and `pos_to_mm` / `mm_to_pos` are exact inverses
// inside each half.
// ---------------------------------------------------------------------------

const STANDOFF_LOW_MIN_MM: f64 = 1.0;
const STANDOFF_LOW_MAX_MM: f64 = 500.0;
const STANDOFF_HIGH_MIN_MM: f64 = 1000.0;
const STANDOFF_HIGH_MAX_MM: f64 = 32000.0;
const STANDOFF_POS_MAX: f64 = 1000.0;
const STANDOFF_POS_SPLIT: f64 = STANDOFF_POS_MAX / 2.0;
const STANDOFF_MIN_MM: f64 = STANDOFF_LOW_MIN_MM;
const STANDOFF_MAX_MM: f64 = STANDOFF_HIGH_MAX_MM;
/// Shared start height (mm) — identical for every algorithm.
const STANDOFF_DEFAULT_MM: f64 = 16000.0;

const MEMORY_LIMIT_BYTES: f64 = 5.5 * 1024.0 * 1024.0 * 1024.0;
const RELOAD_MEMORY_LIMIT_BYTES: f64 = 5.25 * 1024.0 * 1024.0 * 1024.0;

const ALGO_KEY: &str = "ryugu_algo";
const ALGO_ORDER: [&str; 5] = ["werner", "mascon", "rtfp", "carlson", "carlsonalpha"];

const TAB_LEASE_KEY: &str = "ryugu-cauchy-gravity-active-tab-v1";
const TAB_ID_KEY: &str = "ryugu-cauchy-gravity-tab-id-v1";
const TAB_CHANNEL_KEY: &str = "ryugu-cauchy-gravity-tab-channel-v1";
const LEASE_MS: f64 = 3000.0;

// ---------------------------------------------------------------------------
// Observation-height algebra
// ---------------------------------------------------------------------------

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn clamp_standoff(mm: f64) -> f64 {
    if !mm.is_finite() {
        return STANDOFF_DEFAULT_MM;
    }
    mm.clamp(STANDOFF_MIN_MM, STANDOFF_MAX_MM)
}

/// Slider position (0…1000) → observation height in mm.
fn pos_to_mm(pos: f64) -> f64 {
    let p = if pos.is_finite() {
        pos.clamp(0.0, STANDOFF_POS_MAX)
    } else {
        0.0
    };
    if p <= STANDOFF_POS_SPLIT {
        lerp(
            STANDOFF_LOW_MIN_MM,
            STANDOFF_LOW_MAX_MM,
            p / STANDOFF_POS_SPLIT,
        )
    } else {
        lerp(
            STANDOFF_HIGH_MIN_MM,
            STANDOFF_HIGH_MAX_MM,
            (p - STANDOFF_POS_SPLIT) / STANDOFF_POS_SPLIT,
        )
    }
}

/// Observation height in mm → slider position (0…1000).
fn mm_to_pos(mm: f64) -> f64 {
    let v = clamp_standoff(mm);
    if v <= STANDOFF_LOW_MAX_MM {
        return lerp(
            0.0,
            STANDOFF_POS_SPLIT,
            (v - STANDOFF_LOW_MIN_MM) / (STANDOFF_LOW_MAX_MM - STANDOFF_LOW_MIN_MM),
        )
        .round();
    }
    let hi = v.max(STANDOFF_HIGH_MIN_MM);
    lerp(
        STANDOFF_POS_SPLIT,
        STANDOFF_POS_MAX,
        (hi - STANDOFF_HIGH_MIN_MM) / (STANDOFF_HIGH_MAX_MM - STANDOFF_HIGH_MIN_MM),
    )
    .round()
}

fn fmt_mm(mm: f64) -> String {
    if mm >= 1000.0 {
        format!("{:.2} m", mm / 1000.0)
    } else if mm >= 10.0 {
        format!("{mm:.0} mm")
    } else {
        format!("{mm:.1} mm")
    }
}
