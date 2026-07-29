// im2colの逆写像。conv2dのbackwardでgxを求めるために使う。
// 形状の名前はconvのforward基準 (conv_transposeとは逆)。
// x: (group, batch, outShape0, outShape1, chInPerGroup, kernelShape0, kernelShape1)
// y: (batch, group, chInPerGroup, inShape0, inShape1)

@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  len: u32,
  batch: u32,
  group: u32,
  chInPerGroup: u32,
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

@group(0) @binding(2) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let batch = metaBuffer.batch;
  let group = metaBuffer.group;
  let chInPerGroup = metaBuffer.chInPerGroup;
  let inShape1 = metaBuffer.inShape1;
  let outShape0 = i32(metaBuffer.outShape0);
  let outShape1 = i32(metaBuffer.outShape1);
  let kernelShape0 = metaBuffer.kernelShape0;
  let kernelShape1 = metaBuffer.kernelShape1;
  let stride0 = i32(metaBuffer.stride0);
  let stride1 = i32(metaBuffer.stride1);
  let dilation0 = i32(metaBuffer.dilation0);
  let dilation1 = i32(metaBuffer.dilation1);
  for (var j = global_id.x; j < len; j = j + 4096u) {
    var t = j;
    let o1 = i32(t % inShape1);
    t = t / inShape1;
    let o0 = i32(t % metaBuffer.inShape0);
    t = t / metaBuffer.inShape0;
    let c = t % chInPerGroup;
    t = t / chInPerGroup;
    let g = t % group;
    let b = t / group;

    var v = 0.0;
    for (var k0 = 0u; k0 < kernelShape0; k0 = k0 + 1u) {
      let i0s = o0 + i32(metaBuffer.pad0) - i32(k0) * dilation0;
      let i0 = i0s / stride0;
      if (i0s - i0 * stride0 != 0 || i0 < 0 || i0 >= outShape0) {
        continue;
      }
      for (var k1 = 0u; k1 < kernelShape1; k1 = k1 + 1u) {
        let i1s = o1 + i32(metaBuffer.pad1) - i32(k1) * dilation1;
        let i1 = i1s / stride1;
        if (i1s - i1 * stride1 != 0 || i1 < 0 || i1 >= outShape1) {
          continue;
        }
        let xIndex = ((((g * batch + b) * u32(outShape0) + u32(i0)) * u32(outShape1) + u32(i1))
          * chInPerGroup + c) * kernelShape0 * kernelShape1
          + k0 * kernelShape1 + k1;
        v = v + array_x[xIndex];
      }
    }
    array_y[j] = v;
  }
}
