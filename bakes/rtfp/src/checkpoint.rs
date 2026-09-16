//! Temporary per-vertex tensor checkpoint for GPU solvers.
//!
//! RHGF records can only resume once face scalars start appearing. GPU solvers
//! spend most of their time computing vertex tensors before that point, so a
//! separate prefix checkpoint keeps interrupted runs genuinely resumable.

use crate::tensor::Sym6;
use std::fs;
use std::io;
use std::path::Path;

const MAGIC: &[u8; 8] = b"RYVHCP01";
const HEADER: usize = 36;

pub fn load(
    path: &Path,
    expected_vertices: usize,
    expected_standoff_mm: f64,
) -> io::Result<Option<Vec<Sym6>>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if bytes.len() < HEADER || &bytes[0..8] != MAGIC {
        return Ok(None);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != 2 {
        return Ok(None);
    }
    let vertices = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
    let completed = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    let standoff_mm = f64::from_le_bytes(bytes[28..36].try_into().unwrap());
    if vertices != expected_vertices || completed > vertices {
        return Ok(None);
    }
    if !standoff_mm.is_finite() || (standoff_mm - expected_standoff_mm).abs() > 1e-3 {
        return Ok(None);
    }
    let needed = HEADER + completed * 48;
    if bytes.len() < needed {
        return Ok(None);
    }
    let mut tensors = vec![[f64::NAN; 6]; completed];
    for (i, tensor) in tensors.iter_mut().enumerate() {
        for (component, value) in tensor.iter_mut().enumerate() {
            let offset = HEADER + i * 48 + component * 8;
            *value = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        }
    }
    Ok(Some(tensors))
}

pub fn save(path: &Path, tensors: &[Sym6], completed: usize, standoff_mm: f64) -> io::Result<()> {
    let completed = completed.min(tensors.len());
    let mut bytes = Vec::with_capacity(HEADER + completed * 48);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(completed as u64).to_le_bytes());
    bytes.extend_from_slice(&standoff_mm.to_le_bytes());
    for tensor in &tensors[..completed] {
        for value in tensor {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::{load, save};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn checkpoint_is_bound_to_standoff() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ryugu-checkpoint-{}-{unique}.bin",
            std::process::id()
        ));
        let tensors = vec![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];

        save(&path, &tensors, tensors.len(), 16000.0).unwrap();
        assert_eq!(load(&path, 1, 16000.0).unwrap().unwrap(), tensors);
        assert!(load(&path, 1, 8000.0).unwrap().is_none());

        std::fs::remove_file(path).unwrap();
    }
}
