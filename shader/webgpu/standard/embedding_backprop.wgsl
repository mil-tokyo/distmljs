// 同じインデックスが複数回現れうるため、勾配の書き込みは加算になる。
// WGSLのアトミック演算はu32/i32のみのため、compare-exchangeループでf32を加算する。
// array_gwは呼び出し側で0初期化しておくこと。

@group(0) @binding(0) var<storage, read> array_x: array<i32>;
@group(0) @binding(1) var<storage, read> array_gy: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_gw: array<atomic<u32>>;

struct MetaBuffer {
  len: u32,
  embeddingDim: u32,
  numEmbeddings: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

fn atomicAddF32(index: u32, value: f32) {
  var old = atomicLoad(&array_gw[index]);
  loop {
    let updated = bitcast<u32>(bitcast<f32>(old) + value);
    let result = atomicCompareExchangeWeak(&array_gw[index], old, updated);
    if (result.exchanged) {
      break;
    }
    old = result.old_value;
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let embeddingDim = metaBuffer.embeddingDim;
  let numEmbeddings = i32(metaBuffer.numEmbeddings);
  for (var k = global_id.x; k < len; k = k + 4096u) {
    let i = k / embeddingDim;
    let j = k % embeddingDim;
    let e = array_x[i];
    if (e < 0 || e >= numEmbeddings) {
      continue;
    }
    atomicAddF32(u32(e) * embeddingDim + j, array_gy[k]);
  }
}
