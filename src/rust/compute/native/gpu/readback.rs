impl Scene {
    fn read_f32(&self, count: usize) -> Result<Vec<f32>, String> {
        let bytes = count * 4;
        if bytes > self.readback_len {
            return Err("readback buffer too small".into());
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..bytes as u64)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        loop {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|e| format!("device poll: {e}"))?;
            match rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => return Err(format!("readback failed: {e}")),
                Err(std::sync::mpsc::TryRecvError::Empty) => continue,
                Err(e) => return Err(format!("readback channel: {e}")),
            }
        }
        let mapped = self.readback.slice(..bytes as u64).get_mapped_range();
        let mut out = Vec::with_capacity(count);
        for chunk in mapped.as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(*chunk));
        }
        drop(mapped);
        self.readback.unmap();
        Ok(out)
    }
}
