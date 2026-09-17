// ---------------------------------------------------------------------------
// Record helpers (RHGF v5)
// ---------------------------------------------------------------------------

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn set_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn set_f32(bytes: &mut [u8], at: usize, value: f32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn record_face_count(bytes: &[u8]) -> usize {
    if bytes.len() < 12 {
        return 0;
    }
    u32_at(bytes, 8) as usize
}

fn make_record(checkpoint: Option<&[u8]>, height_mm: f64, count: usize) -> Vec<u8> {
    let needed = RECORD_HEADER + count * 4;
    let mut bytes = match checkpoint {
        Some(existing) if existing.len() >= needed => existing.to_vec(),
        _ => vec![0u8; needed],
    };
    let compatible = bytes.len() >= needed
        && u32_at(&bytes, 0) == RECORD_MAGIC
        && u32_at(&bytes, 4) == RECORD_VERSION
        && u32_at(&bytes, 8) as usize == count
        && (f32_at(&bytes, 16) as f64 - height_mm).abs() <= 1e-3;
    if !compatible {
        bytes.fill(0);
    }
    set_u32(&mut bytes, 0, RECORD_MAGIC);
    set_u32(&mut bytes, 4, RECORD_VERSION);
    set_u32(&mut bytes, 8, count as u32);
    set_f32(&mut bytes, 16, height_mm as f32);
    set_f32(&mut bytes, 20, f32::INFINITY);
    set_f32(&mut bytes, 24, f32::NEG_INFINITY);
    bytes
}

fn set_completed(bytes: &mut [u8], completed: usize) {
    let clamped = completed.min(record_face_count(bytes));
    set_u32(bytes, 12, clamped as u32);
}

fn write_scalars(record: &mut [u8], offset: usize, scalars: &[f32]) {
    for (index, value) in scalars.iter().enumerate() {
        set_f32(record, RECORD_HEADER + (offset + index) * 4, *value);
    }
}

