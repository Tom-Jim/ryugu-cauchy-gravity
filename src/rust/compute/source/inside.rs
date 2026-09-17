/// A `+X` ray/triangle intersection, returning the crossing distance or `-1`.
fn ray_triangle_t(origin: Vec3, direction: Vec3, triangle: &Triangle) -> f64 {
    let e1 = sub(triangle.b, triangle.a);
    let e2 = sub(triangle.c, triangle.a);
    let p = cross(direction, e2);
    let determinant = dot(e1, p);
    if determinant.abs() < EPS {
        return -1.0;
    }
    let inverse = 1.0 / determinant;
    let s = sub(origin, triangle.a);
    let u = inverse * dot(s, p);
    if !(0.0..=1.0).contains(&u) {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = inverse * dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return -1.0;
    }
    let t = inverse * dot(e2, q);
    if t > EPS {
        t
    } else {
        -1.0
    }
}

/// Uniform 96³ grid over the body, so an inside test only walks the cells the
/// `+X` ray still has to cross.
struct InsideAccelerator<'a> {
    min: Vec3,
    max: Vec3,
    offsets: Vec<u32>,
    indices: Vec<u32>,
    seen: Vec<i32>,
    stamp: i32,
    triangles: &'a [Triangle],
}

const ACCEL_AXES: usize = 96;

impl<'a> InsideAccelerator<'a> {
    fn new(triangles: &'a [Triangle]) -> Self {
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
        for axis in 0..3 {
            let pad = 1e-3 * (max[axis] - min[axis] + 1.0);
            min[axis] -= pad;
            max[axis] += pad;
        }

        let axes = [ACCEL_AXES, ACCEL_AXES, ACCEL_AXES];
        let cells = ACCEL_AXES * ACCEL_AXES * ACCEL_AXES;
        let mut counts = vec![0u32; cells + 1];
        let mut ranges = vec![0u32; triangles.len() * 6];
        let cell_range = |triangle: &Triangle| -> [u32; 6] {
            let mut lo: Vec3 = [f64::INFINITY; 3];
            let mut hi: Vec3 = [f64::NEG_INFINITY; 3];
            for point in [triangle.a, triangle.b, triangle.c] {
                for axis in 0..3 {
                    lo[axis] = lo[axis].min(point[axis]);
                    hi[axis] = hi[axis].max(point[axis]);
                }
            }
            let mut out = [0u32; 6];
            for axis in 0..3 {
                let extent = max[axis] - min[axis];
                let low = (0.0_f64)
                    .max(((lo[axis] - min[axis]) / extent * axes[axis] as f64).floor())
                    .min(axes[axis] as f64 - 1.0);
                let high = (0.0_f64)
                    .max(((hi[axis] - min[axis]) / extent * axes[axis] as f64).floor())
                    .min(axes[axis] as f64 - 1.0);
                out[axis * 2] = low as u32;
                out[axis * 2 + 1] = high as u32;
            }
            out
        };
        for (index, triangle) in triangles.iter().enumerate() {
            let range = cell_range(triangle);
            ranges[index * 6..index * 6 + 6].copy_from_slice(&range);
            for k in range[4]..=range[5] {
                for j in range[2]..=range[3] {
                    for i in range[0]..=range[1] {
                        counts[(k as usize * ACCEL_AXES + j as usize) * ACCEL_AXES
                            + i as usize
                            + 1] += 1;
                    }
                }
            }
        }
        for i in 0..cells {
            counts[i + 1] += counts[i];
        }
        let offsets = counts;
        let mut indices = vec![0u32; offsets[cells] as usize];
        let mut cursor = vec![0u32; cells];
        for index in 0..triangles.len() {
            let range = &ranges[index * 6..index * 6 + 6];
            for k in range[4]..=range[5] {
                for j in range[2]..=range[3] {
                    for i in range[0]..=range[1] {
                        let cell = (k as usize * ACCEL_AXES + j as usize) * ACCEL_AXES + i as usize;
                        let slot = offsets[cell] as usize + cursor[cell] as usize;
                        indices[slot] = index as u32;
                        cursor[cell] += 1;
                    }
                }
            }
        }
        Self {
            min,
            max,
            offsets,
            indices,
            seen: vec![0i32; triangles.len()],
            stamp: 0,
            triangles,
        }
    }

    fn inside(&mut self, point: Vec3) -> bool {
        let direction = [1.0, 0.0, 0.0];
        let axis_index = |value: f64, axis: usize| -> usize {
            (0.0_f64)
                .max(
                    ((value - self.min[axis]) / (self.max[axis] - self.min[axis])
                        * ACCEL_AXES as f64)
                        .floor(),
                )
                .min(ACCEL_AXES as f64 - 1.0) as usize
        };
        let j = axis_index(point[1], 1);
        let k = axis_index(point[2], 2);
        let first_i = axis_index(point[0], 0);

        self.stamp += 1;
        if self.stamp == 0x7fff_ffff {
            self.seen.fill(0);
            self.stamp = 1;
        }
        let mut hits = 0usize;
        for i in first_i..ACCEL_AXES {
            let cell = (k * ACCEL_AXES + j) * ACCEL_AXES + i;
            for slot in self.offsets[cell]..self.offsets[cell + 1] {
                let triangle = self.indices[slot as usize] as usize;
                if self.seen[triangle] == self.stamp {
                    continue;
                }
                self.seen[triangle] = self.stamp;
                if ray_triangle_t(point, direction, &self.triangles[triangle]) > 0.0 {
                    hits += 1;
                }
            }
        }
        !hits.is_multiple_of(2)
    }
}

