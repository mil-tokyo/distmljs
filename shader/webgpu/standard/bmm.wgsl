// Batched matrix multiply.
// a: (batchSize, M, K) with strideA*, b: (batchSize, K, N) with strideB*,
// y: (batchSize, M, N) contiguous.
// The strides let the caller transpose either operand without materializing it.

@group(0) @binding(0) var<storage, read> array_a: array<f32>;
@group(0) @binding(1) var<storage, read> array_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  M: u32,
  N: u32,
  K: u32,
  strideA0: u32,
  strideA1: u32,
  strideA2: u32,
  strideB0: u32,
  strideB1: u32,
  strideB2: u32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let M = metaBuffer.M;
  let N = metaBuffer.N;
  let K = metaBuffer.K;
  let batch = global_id.z;
  let ofsA = batch * metaBuffer.strideA0;
  let ofsB = batch * metaBuffer.strideB0;
  let ofsY = batch * M * N;
  let strideA1 = metaBuffer.strideA1;
  let strideA2 = metaBuffer.strideA2;
  let strideB1 = metaBuffer.strideB1;
  let strideB2 = metaBuffer.strideB2;
  for (var x = global_id.x; x < N; x = x + 256u) {
    for (var y = global_id.y; y < M; y = y + 256u) {
      var sum = 0.0;
      for (var k = 0u; k < K; k = k + 1u) {
        sum = sum + array_a[ofsA + y * strideA1 + k * strideA2] * array_b[ofsB + k * strideB1 + x * strideB2];
      }
      array_y[ofsY + x + y * N] = sum;
    }
  }
}
