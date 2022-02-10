#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayA {
  float numbers[];
} array_a;

layout(std430, set = 0, binding = 1) readonly buffer arrayB {
  float numbers[];
} array_b;

layout(std430, set = 0, binding = 2) buffer arrayY {
  float numbers[];
} array_y;

layout(std430, set = 0, binding = 3) readonly buffer arrayMeta {
  uint len;
} meta;

void main() {
  uint len = meta.len;
  if (gl_GlobalInvocationID.x != 0) {
    return;
  }
  float v;
  for (uint i = 0; i < len; i++) {
    float diff = array_a.numbers[i] - array_b.numbers[i];
    v += diff * diff;
  }
  v /= float(len);
  array_y.numbers[0] = v;
}
