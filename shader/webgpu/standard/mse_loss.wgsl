@group(0) @binding(0) var<storage, read> array_a: array<f32>;
@group(0) @binding(1) var<storage, read> array_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  len: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  if (global_id.x != 0u) {
    return;
  }
  var v = 0.0;
  for (var i = 0u; i < len; i = i + 1u) {
    let diff = array_a[i] - array_b[i];
    v = v + diff * diff;
  }
  v = v / f32(len);
  array_y[0] = v;
}
