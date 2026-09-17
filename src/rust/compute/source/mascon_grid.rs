fn build_mascon(
    triangles: &[Triangle],
    cauchy_kernels: &[Kernel],
    elliptic_kernels: &[Kernel],
) -> Result<Vec<u8>, String> {
    let mut min: Vec3 = [f64::INFINITY; 3];
    let mut max: Vec3 = [f64::NEG_INFINITY; 3];
    for triangle in triangles {
        for point in [triangle.a, triangle.b, triangle.c] {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
    }
    let shortest = (max[0] - min[0]).min(max[1] - min[1]).min(max[2] - min[2]);
    let pad = 0.5 * shortest / GRID_SIZE as f64;
    for axis in 0..3 {
        min[axis] -= pad;
        max[axis] += pad;
    }
    let extent = sub(max, min);
    let cell: Vec3 = [
        extent[0] / GRID_SIZE as f64,
        extent[1] / GRID_SIZE as f64,
        extent[2] / GRID_SIZE as f64,
    ];
    let volume = cell[0] * cell[1] * cell[2];
    let capacity = (GRID_SIZE * GRID_SIZE * GRID_SIZE).div_ceil(2);
    let mut coords = vec![0u16; capacity * 3];
    let mut cauchy_mass = vec![0.0f32; capacity];
    let mut elliptic_mass = vec![0.0f32; capacity];
    let mut accelerator = InsideAccelerator::new(triangles);
    let mut count = 0usize;
    let mut cauchy_raw = 0.0f64;
    let mut elliptic_raw = 0.0f64;

    for z in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let point: Vec3 = [
                    min[0] + (x as f64 + 0.5) * cell[0],
                    min[1] + (y as f64 + 0.5) * cell[1],
                    min[2] + (z as f64 + 0.5) * cell[2],
                ];
                if !accelerator.inside(point) {
                    continue;
                }
                if count >= capacity {
                    return Err("mascon capacity exceeded".into());
                }
                let cauchy = density_at(cauchy_kernels, point) * volume;
                let elliptic = density_at(elliptic_kernels, point) * volume;
                coords[count * 3] = x as u16;
                coords[count * 3 + 1] = y as u16;
                coords[count * 3 + 2] = z as u16;
                cauchy_mass[count] = cauchy as f32;
                elliptic_mass[count] = elliptic as f32;
                cauchy_raw += cauchy;
                elliptic_raw += elliptic;
                count += 1;
            }
        }
    }

    let cauchy_scale = if CAUCHY_TARGET_MASS > 0.0 && cauchy_raw.abs() > 1e-30 {
        CAUCHY_TARGET_MASS / cauchy_raw
    } else {
        1.0
    };
    if cauchy_scale != 1.0 {
        for mass in cauchy_mass.iter_mut().take(count) {
            *mass *= cauchy_scale as f32;
        }
    }

    let mut out = Out::new(64 + count * 16);
    out.u32(0, 0x314d_5952); // RYM1
    out.u32(4, 1);
    out.u32(8, GRID_SIZE as u32);
    out.u32(12, count as u32);
    out.f32(16, CONSTANT_DENSITY);
    out.f32(20, volume);
    out.f32(24, cauchy_scale);
    out.u32(28, 0);
    for axis in 0..3 {
        out.f32(32 + axis * 4, min[axis]);
        out.f32(44 + axis * 4, max[axis]);
    }
    out.f32(56, cauchy_raw);
    out.f32(60, elliptic_raw);
    let mut offset = 64;
    for index in 0..count {
        out.u16(offset, coords[index * 3]);
        out.u16(offset + 2, coords[index * 3 + 1]);
        out.u16(offset + 4, coords[index * 3 + 2]);
        out.u16(offset + 6, 0);
        out.f32(offset + 8, cauchy_mass[index] as f64);
        out.f32(offset + 12, elliptic_mass[index] as f64);
        offset += 16;
    }
    Ok(out.bytes)
}
