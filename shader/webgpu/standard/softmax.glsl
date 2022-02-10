#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayX {
  float numbers[];
} array_x;

layout(std430, set = 0, binding = 1) buffer arrayY {
  float numbers[];
} array_y;

layout(std430, set = 0, binding = 2) readonly buffer arrayMeta {
  uint shape0, shape1;
} meta;

float get_array_x(uint dim0, uint dim1) {
  return array_x.numbers[dim0 * meta.shape1 + dim1];
}

float get_array_y(uint dim0, uint dim1) {
  return array_y.numbers[dim0 * meta.shape1 + dim1];
}

void set_array_y(float val, uint dim0, uint dim1) {
  array_y.numbers[dim0 * meta.shape1 + dim1] = val;
}

void main() {
  uint shape0 = meta.shape0, shape1 = meta.shape1;
  for (uint i = gl_GlobalInvocationID.x; i < shape0; i += 4096) {
    float top = 0.0;
    for (uint j = 0; j < shape1; j++) {
      float v = get_array_x(i, j);
      if (v > top) {
        top = v;
      }
    }
    float expsum = 0.0;
    for (uint j = 0; j < shape1; j++) {
      float v = get_array_x(i, j);
      float e = exp(v - top);
      set_array_y(e, i, j);
      expsum += e;
    }
    float inv_expsum = 1.0 / expsum;
    for (uint j = 0; j < shape1; j++) {
      float v = get_array_y(i, j);
      set_array_y(v * inv_expsum, i, j);
    }
  }
}
