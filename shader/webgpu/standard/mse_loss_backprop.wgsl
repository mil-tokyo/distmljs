@group(0) @binding(0) var<storage, read> array_a: array<f32>;
@group(0) @binding(1) var<storage, read> array_b: array<f32>;
@group(0) @binding(2) var<storage, read> array_gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> array_ga: array<f32>;
@group(0) @binding(4) var<storage, read_write> array_gb: array<f32>;

struct MetaBuffer {
  len: u32,
  coef: f32,
}

@group(0) @binding(5) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let coef = metaBuffer.coef;
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let v = (array_a[i] - array_b[i]) * array_gy[0] * coef;
    array_ga[i] = v;
    array_gb[i] = -v;
  }
}
