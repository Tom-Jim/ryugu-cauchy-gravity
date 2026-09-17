//! RHGF v5 progressive face record: the format every algorithm writes and the
//! viewer reads (`magic, version, n_faces, n_done, standoff_mm, s_min, s_max`
//! header, then one f32 per face, NaN = not yet computed).

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const MAGIC: u32 = 0x5248_4746;
pub const VERSION: u32 = 5;
pub const HEADER: usize = 28;

pub struct Record {
    file: File,
    pub n_faces: usize,
    pub scalars: Vec<f32>,
    pub standoff_mm: f32,
    pub s_min: f32,
    pub s_max: f32,
    pub n_done: usize,
}

impl Record {
    /// Truncate/create the record and fill it with NaN.
    pub fn create(path: &Path, n_faces: usize, standoff_mm: f32) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let record = Record {
            file: file.try_clone()?,
            n_faces,
            scalars: vec![f32::NAN; n_faces],
            standoff_mm,
            s_min: f32::INFINITY,
            s_max: f32::NEG_INFINITY,
            n_done: 0,
        };
        write_header(
            &mut file,
            n_faces,
            0,
            standoff_mm,
            f32::INFINITY,
            f32::NEG_INFINITY,
        )?;
        file.write_all(bytemuck_cast(&record.scalars))?;
        file.flush()?;
        Ok(Record { file, ..record })
    }

    /// Reopen an existing record; returns `None` when it is missing or does not
    /// describe this mesh.
    pub fn open_resume(path: &Path, n_faces: usize) -> io::Result<Option<Self>> {
        let Ok(mut file) = OpenOptions::new().read(true).write(true).open(path) else {
            return Ok(None);
        };
        let mut head = [0u8; HEADER];
        if file.read_exact(&mut head).is_err() {
            return Ok(None);
        }
        let magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(head[4..8].try_into().unwrap());
        let n = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let standoff_mm = f32::from_le_bytes(head[16..20].try_into().unwrap());
        if magic != MAGIC || version != VERSION || n != n_faces {
            return Ok(None);
        }
        let mut scalars = vec![f32::NAN; n_faces];
        if file.read_exact(bytemuck_cast_mut(&mut scalars)).is_err() {
            return Ok(None);
        }
        let mut s_min = f32::INFINITY;
        let mut s_max = f32::NEG_INFINITY;
        let mut n_done = 0usize;
        for s in &scalars {
            if s.is_finite() {
                n_done += 1;
                s_min = s_min.min(*s);
                s_max = s_max.max(*s);
            }
        }
        Ok(Some(Record {
            file,
            n_faces,
            scalars,
            standoff_mm,
            s_min,
            s_max,
            n_done,
        }))
    }

    pub fn set_face(&mut self, f: usize, s: f32) -> io::Result<()> {
        self.scalars[f] = s;
        if s.is_finite() {
            self.s_min = self.s_min.min(s);
            self.s_max = self.s_max.max(s);
            self.n_done += 1;
        }
        let off = (HEADER + f * 4) as u64;
        self.file.seek(SeekFrom::Start(off))?;
        self.file.write_all(&s.to_le_bytes())
    }

    pub fn flush_header(&mut self) -> io::Result<()> {
        let mut file = self.file.try_clone()?;
        write_header(
            &mut file,
            self.n_faces,
            self.n_done,
            self.standoff_mm,
            self.s_min,
            self.s_max,
        )?;
        self.file.flush()
    }
}

fn write_header(
    file: &mut File,
    n_faces: usize,
    n_done: usize,
    standoff_mm: f32,
    s_min: f32,
    s_max: f32,
) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&MAGIC.to_le_bytes())?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(n_faces as u32).to_le_bytes())?;
    file.write_all(&(n_done as u32).to_le_bytes())?;
    file.write_all(&standoff_mm.to_le_bytes())?;
    file.write_all(&s_min.to_le_bytes())?;
    file.write_all(&s_max.to_le_bytes())
}

/// Face evaluation order, stored so a resumed run keeps filling the same
/// progressive picture the viewer already started drawing.
pub fn write_order(path: &Path, order: &[u32]) -> io::Result<()> {
    let mut out = File::create(path)?;
    out.write_all(&(order.len() as u32).to_le_bytes())?;
    out.write_all(bytemuck_cast(order))
}

pub fn load_order(path: &Path, n_faces: usize) -> io::Result<Option<Vec<u32>>> {
    let Ok(mut f) = File::open(path) else {
        return Ok(None);
    };
    let mut head = [0u8; 4];
    if f.read_exact(&mut head).is_err() {
        return Ok(None);
    }
    let n = u32::from_le_bytes(head) as usize;
    if n != n_faces {
        return Ok(None);
    }
    let mut order = vec![0u32; n_faces];
    if f.read_exact(bytemuck_cast_mut(&mut order)).is_err() {
        return Ok(None);
    }
    let mut seen = vec![false; n_faces];
    for &f in &order {
        let i = f as usize;
        if i >= n_faces || seen[i] {
            return Ok(None);
        }
        seen[i] = true;
    }
    Ok(Some(order))
}

/// `&[f32]`/`&[u32]` -> bytes without pulling in a dependency for six lines.
fn bytemuck_cast<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_cast_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
