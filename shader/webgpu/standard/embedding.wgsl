// x: (...) のint32インデックス、weight: (numEmbeddings, embeddingDim)
// y: (..., embeddingDim)

@group(0) @binding(0) var<storage, read> array_x: array<i32>;
@group(0) @binding(1) var<storage, read> array_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  len: u32,
  embeddingDim: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let len = metaBuffer.len;
  let embeddingDim = metaBuffer.embeddingDim;
  for (var k = global_id.x; k < len; k = k + 4096u) {
    let i = k / embeddingDim;
    let j = k % embeddingDim;
    let e = array_x[i];
    array_y[k] = array_w[u32(e) * embeddingDim + j];
  }
}
