#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arraySoftmax {
  float numbers[];
} array_softmax;

layout(std430, set = 0, binding = 1) readonly buffer arrayLabel {
  int numbers[];
} array_label;

layout(std430, set = 0, binding = 2) readonly buffer arrayGy {
  float numbers[];
} array_gy;

layout(std430, set = 0, binding = 3) buffer arrayGx {
  float numbers[];
} array_gx;

layout(std430, set = 0, binding = 4) readonly buffer arrayMeta {
  uint len;
  uint shape0, shape1;
} meta;

float get_array_softmax(uint dim0, uint dim1) {
  return array_softmax.numbers[dim0 * meta.shape1 + dim1];
}

int get_array_label(uint dim0) {
  return array_label.numbers[dim0];
}

float get_array_gy(uint dim0, uint dim1) {
  return array_gy.numbers[dim0 * meta.shape1 + dim1];
}

void set_array_gx(float val, uint dim0, uint dim1) {
  array_gx.numbers[dim0 * meta.shape1 + dim1] = val;
}

void main() {
  uint len = meta.len;
  uint shape0 = meta.shape0, shape1 = meta.shape1;
  for (uint i = gl_GlobalInvocationID.x; i < len; i += 4096) {
    uint dim1 = i % shape1;
    uint dim0 = i / shape1;

    float v = get_array_softmax(dim0, dim1);
    int label = get_array_label(dim0);
    if (uint(label) == dim1) {
      v -= 1.0;
    }
    v *= get_array_gy(dim0, dim1) / float(shape0);
    set_array_gx(v, dim0, dim1);
  }
}
