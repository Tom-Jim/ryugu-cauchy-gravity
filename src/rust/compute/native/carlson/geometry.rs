fn face_centre(triangle: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
        (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
        (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0,
    ]
}

fn lerp(origin: [f64; 3], target: [f64; 3], scale: f64) -> [f64; 3] {
    [
        origin[0] + scale * (target[0] - origin[0]),
        origin[1] + scale * (target[1] - origin[1]),
        origin[2] + scale * (target[2] - origin[2]),
    ]
}

fn frustum_centre(origin: [f64; 3], triangle: [[f64; 3]; 3], lo: f64, hi: f64) -> [f64; 3] {
    let mut centre = [0.0f64; 3];
    let mut count = 0.0f64;
    for vertex in triangle {
        for scale in [lo.max(1e-6), hi] {
            let point = lerp(origin, vertex, scale);
            for axis in 0..3 {
                centre[axis] += point[axis];
            }
            count += 1.0;
        }
    }
    [centre[0] / count, centre[1] / count, centre[2] / count]
}

fn push_face(
    out: &mut Vec<Face>,
    p: [f64; 3],
    q: [f64; 3],
    r: [f64; 3],
    inside: [f64; 3],
    weight: f64,
) {
    let (mut q, mut r) = (q, r);
    let mut normal = cross(sub(q, p), sub(r, p));
    let centre = [
        (p[0] + q[0] + r[0]) / 3.0,
        (p[1] + q[1] + r[1]) / 3.0,
        (p[2] + q[2] + r[2]) / 3.0,
    ];
    if dot(normal, sub(centre, inside)) < 0.0 {
        std::mem::swap(&mut q, &mut r);
        normal = cross(sub(q, p), sub(r, p));
    }
    let length = norm(normal);
    let unit = if length > 1e-30 {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    } else {
        [0.0, 0.0, 1.0]
    };
    out.push(Face {
        corners: [p, q, r],
        n: unit,
        weight,
    });
}

fn tet_signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
