/// Little-endian writer over a fixed-size runtime asset buffer.
struct Out {
    bytes: Vec<u8>,
}

impl Out {
    fn new(len: usize) -> Self {
        Self {
            bytes: vec![0u8; len],
        }
    }

    fn u16(&mut self, at: usize, value: u16) {
        self.bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, at: usize, value: u32) {
        self.bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, at: usize, value: f64) {
        self.bytes[at..at + 4].copy_from_slice(&(value as f32).to_le_bytes());
    }
}
