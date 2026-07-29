@group(0) @binding(0) var<storage, read> array_softmax: array<f32>;
@group(0) @binding(1) var<storage, read> array_label: array<i32>;
@group(0) @binding(2) var<storage, read> array_gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> array_gx: array<f32>;

struct MetaBuffer {
  len: u32,
  shape0: u32,
  shape1: u32,
}

@group(0) @binding(4) var<storage, read> metaBuffer: MetaBuffer;

fn get_array_softmax(dim0: u32, dim1: u32) -> f32 {
  return array_softmax[dim0 * metaBuffer.shape1 + dim1];
}

fn get_array_label(dim0: u32) -> i32 {
  return array_label[dim0];
}

fn get_array_gy(dim0: u32, dim1: u32) -> f32 {
  return array_gy[dim0 * metaBuffer.shape1 + dim1];
}

fn set_array_gx(val: f32, dim0: u32, dim1: u32) {
  array_gx[dim0 * metaBuffer.shape1 + dim1] = val;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let shape0 = metaBuffer.shape0;
  let shape1 = metaBuffer.shape1;
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let dim1 = i % shape1;
    let dim0 = i / shape1;

    var v = get_array_softmax(dim0, dim1);
    let labelValue = get_array_label(dim0);
    if (u32(labelValue) == dim1) {
      v = v - 1.0;
    }
    v = v * get_array_gy(dim0, dim1) / f32(shape0);
    set_array_gx(v, dim0, dim1);
  }
}
