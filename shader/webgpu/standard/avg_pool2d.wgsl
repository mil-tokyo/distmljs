// x: (batch, ch, inShape0, inShape1), y: (batch, ch, outShape0, outShape1)
// divMode 0: divisorOverrideによる定数areaMul、1: パディングを数に含める、2: 含めない

@group(0) @binding(0) var<storage, read> array_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> array_y: array<f32>;

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
  pad0Begin: u32,
  pad1Begin: u32,
  pad0End: u32,
  pad1End: u32,
  divMode: u32,
  areaMul: f32,
}

@group(0) @binding(2) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let inShape0 = i32(metaBuffer.inShape0);
  let inShape1 = i32(metaBuffer.inShape1);
  let outShape0 = metaBuffer.outShape0;
  let outShape1 = metaBuffer.outShape1;
  let divMode = metaBuffer.divMode;
  let inSpLen = u32(inShape0 * inShape1);
  let padEnd0 = inShape0 + i32(metaBuffer.pad0End);
  let padEnd1 = inShape1 + i32(metaBuffer.pad1End);
  for (var i = global_id.x; i < len; i = i + 4096u) {
    let o1 = i % outShape1;
    let o0 = (i / outShape1) % outShape0;
    let bc = i / (outShape0 * outShape1);

    var v = 0.0;
    var area = 0.0;
    for (var k0 = 0u; k0 < metaBuffer.kernelShape0; k0 = k0 + 1u) {
      let in0 = i32(o0 * metaBuffer.stride0 + k0) - i32(metaBuffer.pad0Begin);
      for (var k1 = 0u; k1 < metaBuffer.kernelShape1; k1 = k1 + 1u) {
        let in1 = i32(o1 * metaBuffer.stride1 + k1) - i32(metaBuffer.pad1Begin);
        if (in0 >= 0 && in0 < inShape0 && in1 >= 0 && in1 < inShape1) {
          v = v + array_x[bc * inSpLen + u32(in0 * inShape1 + in1)];
          if (divMode == 2u) {
            area = area + 1.0;
          }
        }
        if (divMode == 1u && in0 < padEnd0 && in1 < padEnd1) {
          area = area + 1.0;
        }
      }
    }
    if (divMode == 0u) {
      v = v * metaBuffer.areaMul;
    } else {
      v = v / area;
    }
    array_y[i] = v;
  }
}
