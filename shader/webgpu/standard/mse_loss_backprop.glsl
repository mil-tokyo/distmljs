#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayA {
  float numbers[];
} array_a;

layout(std430, set = 0, binding = 1) readonly buffer arrayB {
  float numbers[];
} array_b;

layout(std430, set = 0, binding = 2) readonly buffer arrayGy {
  float numbers[];
} array_gy;

layout(std430, set = 0, binding = 3) buffer arrayGa {
  float numbers[];
} array_ga;

layout(std430, set = 0, binding = 4) buffer arrayGb {
  float numbers[];
} array_gb;

layout(std430, set = 0, binding = 5) readonly buffer arrayMeta {
  uint len;
  float coef;
} meta;

float get_array_a(uint dim0) {
  return array_a.numbers[dim0];
}

float get_array_b(uint dim0) {
  return array_b.numbers[dim0];
}

float get_array_gy() {
  return array_gy.numbers[0];
}

void set_array_ga(float val, uint dim0) {
  array_ga.numbers[dim0] = val;
}

void set_array_gb(float val, uint dim0) {
  array_gb.numbers[dim0] = val;
}

void main() {
  uint len = meta.len;
  float coef = meta.coef;
  for (uint i = gl_GlobalInvocationID.x; i < len; i += 4096) {
    float v = (get_array_a(i) - get_array_b(i)) * get_array_gy() * coef;
    set_array_ga(v, i);
    set_array_gb(-v, i);
  }
}
