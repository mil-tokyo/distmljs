// 1つの軸に沿った最大値・最小値とそのインデックスを求める。
// 入力は連続なバッファで、縮約する軸をredLength、その内側の軸の積をinnerLengthとする。
// isMaxが0以外なら最大値、0なら最小値。
// CPU実装と同じく比較は狭義の不等号で、同値の場合は先に現れたインデックスを返す。

@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_indices: array<i32>;

struct MetaBuffer {
  len: u32,
  redLength: u32,
  innerLength: u32,
  isMax: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let redLength = metaBuffer.redLength;
  let innerLength = metaBuffer.innerLength;
  let isMax = metaBuffer.isMax != 0u;
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let outer = i / innerLength;
    let inner = i % innerLength;
    let base = outer * redLength * innerLength + inner;
    var best = array_x[base];
    var bestIndex = 0u;
    for (var j = 1u; j < redLength; j = j + 1u) {
      let v = array_x[base + j * innerLength];
      let better = select(best > v, best < v, isMax);
      if (better) {
        best = v;
        bestIndex = j;
      }
    }
    array_values[i] = best;
    array_indices[i] = i32(bestIndex);
  }
}
