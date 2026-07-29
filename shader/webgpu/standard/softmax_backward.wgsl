// softmaxの出力yと出力側の勾配gyから入力側の勾配を求める。
// y / gy / gx はいずれも (batch, cs) として扱う。

@group(0) @binding(0) var<storage, read> array_y: array<f32>;
@group(0) @binding(1) var<storage, read> array_gy: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_gx: array<f32>;

struct MetaBuffer {
  len: u32,
  cs: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let cs = metaBuffer.cs;
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let c = i % cs;
    let base = i - c;
    let my = array_y[i];
    var sum = 0.0;
    for (var d = 0u; d < cs; d = d + 1u) {
      if (d == c) {
        sum = sum + my * (1.0 - my) * array_gy[base + d];
      } else {
        sum = sum - my * array_y[base + d] * array_gy[base + d];
      }
    }
    array_gx[i] = sum;
  }
}
