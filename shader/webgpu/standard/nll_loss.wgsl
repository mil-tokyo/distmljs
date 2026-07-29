@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read> array_label: array<i32>;
@group(0) @binding(2) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  shape0: u32,
  shape1: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

fn get_array_x(dim0: u32, dim1: u32) -> f32 {
  return array_x[dim0 * metaBuffer.shape1 + dim1];
}

fn get_array_label(dim0: u32) -> i32 {
  return array_label[dim0];
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x != 0u) {
    return;
  }
  let shape0 = metaBuffer.shape0;
  var v = 0.0;
  for (var i = 0u; i < shape0; i = i + 1u) {
    v = v + log(get_array_x(i, u32(get_array_label(i))));
  }
  v = v / -f32(shape0);
  array_y[0] = v;
}
