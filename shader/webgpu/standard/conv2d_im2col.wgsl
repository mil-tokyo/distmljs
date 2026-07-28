// x: (batch, group, chInPerGroup, inShape0, inShape1)
// y: (group, batch, outShape0, outShape1, chInPerGroup, kernelShape0, kernelShape1)

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
  let inShape0 = metaBuffer.inShape0;
  let inShape1 = metaBuffer.inShape1;
  let outShape0 = metaBuffer.outShape0;
  let outShape1 = metaBuffer.outShape1;
  let kernelShape0 = metaBuffer.kernelShape0;
  let kernelShape1 = metaBuffer.kernelShape1;
  for (var i = global_id.x; i < len; i = i + 4096u) {
    var t = i;
    let k1 = t % kernelShape1;
    t = t / kernelShape1;
    let k0 = t % kernelShape0;
    t = t / kernelShape0;
    let ci = t % chInPerGroup;
    t = t / chInPerGroup;
    let o1 = t % outShape1;
    t = t / outShape1;
    let o0 = t % outShape0;
    t = t / outShape0;
    let b = t % batch;
    let g = t / batch;

    let in0 = i32(o0 * metaBuffer.stride0 + k0 * metaBuffer.dilation0) - i32(metaBuffer.pad0);
    let in1 = i32(o1 * metaBuffer.stride1 + k1 * metaBuffer.dilation1) - i32(metaBuffer.pad1);
    var v = 0.0;
    if (in0 >= 0 && in0 < i32(inShape0) && in1 >= 0 && in1 < i32(inShape1)) {
      let xIndex = ((b * group + g) * chInPerGroup + ci) * inShape0 * inShape1
        + u32(in0) * inShape1 + u32(in1);
      v = array_x[xIndex];
    }
    array_y[i] = v;
  }
}
