#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayX {
  float numbers[];
} array_x;

layout(std430, set = 0, binding = 1) readonly buffer arrayLabel {
  int numbers[];
} array_label;

layout(std430, set = 0, binding = 2) buffer arrayY {
  float numbers[];
} array_y;

layout(std430, set = 0, binding = 3) readonly buffer arrayMeta {
  uint shape0, shape1;
} meta;

float get_array_x(uint dim0, uint dim1) {
  return array_x.numbers[dim0 * meta.shape1 + dim1];
}

int get_array_label(uint dim0) {
  return array_label.numbers[dim0];
}

void main() {
  if (gl_GlobalInvocationID.x != 0) {
    return;
  }
  uint shape0 = meta.shape0, shape1 = meta.shape1;
  float v;
  for (uint i = 0; i < shape0; i++) {
    v += log(get_array_x(i, uint(get_array_label(i))));
  }
  v /= -float(shape0);
  array_y.numbers[0] = v;
}
