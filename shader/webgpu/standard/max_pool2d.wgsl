// x: (batch, ch, inShape0, inShape1), y / indices: (batch, ch, outShape0, outShape1)
// indicesは画像内の平坦化した位置 (in0 * inShape1 + in1)。
// WebGLと異なり1つのカーネルで値とインデックスの両方を出力できる。

@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_indices: array<i32>;

struct MetaBuffer {
  len: u32,
  inShape0: u32,
  inShape1: u32,
  outShape0: u32,
  outShape1: u32,
  kernelShape0: u32,
  kernelShape1: u32,
  stride0: u32,
  stride1: u32,
  pad0: u32,
  pad1: u32,
  dilation0: u32,
  dilation1: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let inShape0 = i32(metaBuffer.inShape0);
  let inShape1 = i32(metaBuffer.inShape1);
  let outShape0 = metaBuffer.outShape0;
  let outShape1 = metaBuffer.outShape1;
  let inSpLen = u32(inShape0 * inShape1);
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let o1 = i % outShape1;
    let o0 = (i / outShape1) % outShape0;
    // バッチとチャンネルはまとめて扱う
    let bc = i / (outShape0 * outShape1);

    // CPU実装は-Infinityで初期化するが、WGSLは無限大のリテラルを書けないため、
    // 最初の有効な要素で初期化する。CPU実装と同じく比較は狭義の不等号。
    var best = 0.0;
    var bestIndex = 0;
    var found = false;
    for (var k0 = 0u; k0 < metaBuffer.kernelShape0; k0 = k0 + 1u) {
      let in0 = i32(o0 * metaBuffer.stride0 + k0 * metaBuffer.dilation0) - i32(metaBuffer.pad0);
      if (in0 < 0 || in0 >= inShape0) {
        continue;
      }
      for (var k1 = 0u; k1 < metaBuffer.kernelShape1; k1 = k1 + 1u) {
        let in1 = i32(o1 * metaBuffer.stride1 + k1 * metaBuffer.dilation1) - i32(metaBuffer.pad1);
        if (in1 < 0 || in1 >= inShape1) {
          continue;
        }
        let spatial = in0 * inShape1 + in1;
        let v = array_x[bc * inSpLen + u32(spatial)];
        if (!found || v > best) {
          best = v;
          bestIndex = spatial;
          found = true;
        }
      }
    }
    array_y[i] = best;
    array_indices[i] = bestIndex;
  }
}
