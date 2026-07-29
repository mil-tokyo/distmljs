@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  shape0: u32,
  shape1: u32,
}

@group(0) @binding(2) var<storage, read> metaBuffer: MetaBuffer;

fn get_array_x(dim0: u32, dim1: u32) -> f32 {
  return array_x[dim0 * metaBuffer.shape1 + dim1];
}

fn get_array_y(dim0: u32, dim1: u32) -> f32 {
  return array_y[dim0 * metaBuffer.shape1 + dim1];
}

fn set_array_y(val: f32, dim0: u32, dim1: u32) {
  array_y[dim0 * metaBuffer.shape1 + dim1] = val;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let shape0 = metaBuffer.shape0;
  let shape1 = metaBuffer.shape1;
  for (var i = global_id.x; i < shape0; i = i + 4096u) {
    var top = 0.0;
    for (var j = 0u; j < shape1; j = j + 1u) {
      let v = get_array_x(i, j);
      if (v > top) {
        top = v;
      }
    }
    var expsum = 0.0;
    for (var j = 0u; j < shape1; j = j + 1u) {
      let v = get_array_x(i, j);
      let e = exp(v - top);
      set_array_y(e, i, j);
      expsum = expsum + e;
    }
    let inv_expsum = 1.0 / expsum;
    for (var j = 0u; j < shape1; j = j + 1u) {
      let v = get_array_y(i, j);
      set_array_y(v * inv_expsum, i, j);
    }
  }
}
