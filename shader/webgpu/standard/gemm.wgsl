@group(0) @binding(0) var<storage, read> array_a: array<f32>;
@group(0) @binding(1) var<storage, read> array_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> array_y: array<f32>;

struct MetaBuffer {
  M: u32,
  N: u32,
  K: u32,
  strideA0: u32,
  strideA1: u32,
  strideB0: u32,
  strideB1: u32,
  alpha: f32,
}

@group(0) @binding(3) var<storage, read> metaBuffer: MetaBuffer;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let M = metaBuffer.M;
  let N = metaBuffer.N;
  let K = metaBuffer.K;
  let strideA0 = metaBuffer.strideA0;
  let strideA1 = metaBuffer.strideA1;
  let strideB0 = metaBuffer.strideB0;
  let strideB1 = metaBuffer.strideB1;
  let alpha = metaBuffer.alpha;
  for (var x = global_id.x; x < N; x = x + 256u) {
    for (var y = global_id.y; y < M; y = y + 256u) {
      var sum = 0.0;
      for (var k = 0u; k < K; k = k + 1u) {
        sum = sum + array_a[y * strideA0 + k * strideA1] * array_b[k * strideB0 + x * strideB1];
      }
      array_y[x + y * N] = sum * alpha;
    }
  }
}
