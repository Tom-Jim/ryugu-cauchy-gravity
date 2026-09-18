// ---------------------------------------------------------------------------
// RHGF v5 record reader
// ---------------------------------------------------------------------------

const RECORD_MAGIC: u32 = 0x5248_4746;
const RECORD_VERSION: u32 = 5;
const RECORD_HEADER: usize = 28;

#[derive(Clone, Debug, Default, PartialEq)]
struct ParsedRecord {
    scalars: Vec<f32>,
    standoff_mm: f64,
    total: usize,
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(raw)
}

fn read_f32(bytes: &[u8], at: usize) -> f32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[at..at + 4]);
    f32::from_le_bytes(raw)
}

fn record_face_count(bytes: &[u8]) -> usize {
    if bytes.len() < 12 {
        return 0;
    }
    read_u32(bytes, 8) as usize
}

fn parse_face_scalars(bytes: &[u8]) -> Option<ParsedRecord> {
    if bytes.len() < RECORD_HEADER {
        return None;
    }
    if read_u32(bytes, 0) != RECORD_MAGIC {
        return None;
    }
    // v5 keeps the scalars at 28 + 4·total; an older header would shift every
    // value by one float and silently produce a bogus comparison.
    if read_u32(bytes, 4) != RECORD_VERSION {
        return None;
    }
    let total = read_u32(bytes, 8) as usize;
    if bytes.len() < RECORD_HEADER + total * 4 {
        return None;
    }
    // Offset 16 is the reserved slot holding the observation height (mm).
    let standoff_mm = read_f32(bytes, 16) as f64;
    let scalars = (0..total)
        .map(|index| read_f32(bytes, RECORD_HEADER + index * 4))
        .collect();
    Some(ParsedRecord {
        scalars,
        standoff_mm,
        total,
    })
}

/// True when a record carries at least one non-zero finite face scalar.
///
/// A Frobenius norm of a real gravity-gradient field is strictly positive, so an
/// all-zero record proves the kernels never ran. A WebGPU pipeline that failed
/// validation used to be dropped silently and still produce exactly that, so
/// this is the last guard before a broken run is painted.
fn record_has_signal(parsed: &ParsedRecord) -> bool {
    if parsed.total == 0 {
        return false;
    }
    parsed
        .scalars
        .iter()
        .any(|value| value.is_finite() && *value != 0.0)
}
