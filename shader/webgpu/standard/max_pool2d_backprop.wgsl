// indicesは画像内の平坦化した位置なので、batchとchをまとめた3次元として扱う。
// gy / indices: (batch * ch, outSpLen), gx: (batch * ch, inSpLen)
// 計算量的には非効率。kernelSizeから探索範囲を絞ったほうが効率的。

@group(0) @binding(0) var<storage, read> array_gy: array<f32>;
@group(0) @binding(1) var<storage, read> array_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> array_gx: array<f32>;

struct MetaBuffer {
  len: u32,
  inSpLen: u32,
  outSpLen: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let inSpLen = metaBuffer.inSpLen;
  let outSpLen = metaBuffer.outSpLen;
  for (var j = global_id.x; j < len; j = j + 4096u) {
    let isp = i32(j % inSpLen);
    let bc = j / inSpLen;
    var v = 0.0;
    for (var i = 0u; i < outSpLen; i = i + 1u) {
      if (array_indices[bc * outSpLen + i] == isp) {
        v = v + array_gy[bc * outSpLen + i];
      }
    }
    array_gx[j] = v;
  }
}
