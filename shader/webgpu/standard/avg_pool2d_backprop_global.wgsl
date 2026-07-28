// global average pooling専用。gyの各要素を入力の全画素に等分に配る。
// gy: (batch * ch), gx: (batch * ch, inSpLen)

@group(0) @binding(0) var<storage, read> array_gy: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_gx: array<f32>;

struct MetaBuffer {
  len: u32,
  inSpLen: u32,
  areaMul: f32,
}

@group(0) @binding(2) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let inSpLen = metaBuffer.inSpLen;
  for (var j = global_id.x; j < len; j = j + 4096u) {
    array_gx[j] = array_gy[j / inSpLen] * metaBuffer.areaMul;
  }
}
