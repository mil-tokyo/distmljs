import { DType, DTypeDefault } from '../../../dtype';
import { WebGLUniformItem } from '../webglContext';
import { TensorTextureShapeDim, WebGLTensor } from '../webglTensor';

export const webglShaderHeader = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp isampler2D;
precision highp isampler2DArray;
precision highp usampler2D;
precision highp usampler2DArray;
`;

export function getTypeForDType(dtype: DType, vec4?: boolean) {
  let scalarType: string;
  let vec4Type: string;
  let samplerPrefix: string;
  switch (dtype) {
    case 'float32':
      samplerPrefix = '';
      scalarType = 'float';
      vec4Type = 'vec4';
      break;
    case 'int32':
      samplerPrefix = 'i';
      scalarType = 'int';
      vec4Type = 'ivec4';
      break;
    case 'uint8':
    case 'bool':
      samplerPrefix = 'u';
      scalarType = 'uint';
      vec4Type = 'uvec4';
      break;
    default:
      throw new Error();
  }

  return {
    scalarType,
    vec4Type,
    samplerPrefix,
    ioType: vec4 ? vec4Type : scalarType,
  };
}

export function shaderGenOutput(
  expr: string,
  dtype: DType = DTypeDefault,
  vec4?: boolean
): string {
  if (vec4) {
    return `fragColor = (${expr});`;
  } else {
    const { samplerPrefix } = getTypeForDType(dtype, false);
    return `fragColor = ${samplerPrefix}vec4((${expr}), 0.0, 0.0, 0.0);`;
  }
}

function shaderGenTensorNDGetInternal(
  name: string,
  ndim: number,
  dim: TensorTextureShapeDim,
  dtype: DType,
  vec4: boolean
): string {
  let args: string, flat_index: string, uniforms: string;
  switch (ndim) {
    case 0:
      uniforms = '';
      args = '';
      flat_index = '0';
      break;
    case 1:
      uniforms = `
uniform int _ka_${name}_stride_0;
          `;
      args = 'int d0';
      flat_index = `d0 * _ka_${name}_stride_0`;
      break;
    case 2:
      uniforms = `
uniform int _ka_${name}_stride_0;
uniform int _ka_${name}_stride_1;
            `;
      args = 'int d0, int d1';
      flat_index = `d0 * _ka_${name}_stride_0 + d1 * _ka_${name}_stride_1`;
      break;
    case 3:
      uniforms = `
uniform int _ka_${name}_stride_0;
uniform int _ka_${name}_stride_1;
uniform int _ka_${name}_stride_2;
              `;
      args = 'int d0, int d1, int d2';
      flat_index = `d0 * _ka_${name}_stride_0 + d1 * _ka_${name}_stride_1 + d2 * _ka_${name}_stride_2`;
      break;
    case 4:
      uniforms = `
uniform int _ka_${name}_stride_0;
uniform int _ka_${name}_stride_1;
uniform int _ka_${name}_stride_2;
uniform int _ka_${name}_stride_3;
        `;
      args = 'int d0, int d1, int d2, int d3';
      flat_index = `d0 * _ka_${name}_stride_0 + d1 * _ka_${name}_stride_1 + d2 * _ka_${name}_stride_2 + d3 * _ka_${name}_stride_3`;
      break;
    case 5:
      uniforms = `
uniform int _ka_${name}_stride_0;
uniform int _ka_${name}_stride_1;
uniform int _ka_${name}_stride_2;
uniform int _ka_${name}_stride_3;
uniform int _ka_${name}_stride_4;
          `;
      args = 'int d0, int d1, int d2, int d3, int d4';
      flat_index = `d0 * _ka_${name}_stride_0 + d1 * _ka_${name}_stride_1 + d2 * _ka_${name}_stride_2 + d3 * _ka_${name}_stride_3 + d4 * _ka_${name}_stride_4`;
      break;
    case 6:
      uniforms = `
uniform int _ka_${name}_stride_0;
uniform int _ka_${name}_stride_1;
uniform int _ka_${name}_stride_2;
uniform int _ka_${name}_stride_3;
uniform int _ka_${name}_stride_4;
uniform int _ka_${name}_stride_5;
          `;
      args = 'int d0, int d1, int d2, int d3, int d4, int d5';
      flat_index = `d0 * _ka_${name}_stride_0 + d1 * _ka_${name}_stride_1 + d2 * _ka_${name}_stride_2 + d3 * _ka_${name}_stride_3 + d4 * _ka_${name}_stride_4 + d5 * _ka_${name}_stride_5`;
      break;
    default:
      throw new Error();
  }
  const { samplerPrefix, ioType } = getTypeForDType(dtype, vec4);
  if (dim === '2D') {
    return `
uniform ${samplerPrefix}sampler2D ${name};
${uniforms}

${ioType} get_${name}(${args}) {
int flat_index = ${flat_index};
int texture_w = textureSize(${name}, 0).x;
int y = flat_index / texture_w;
int x = flat_index - y * texture_w;
return texelFetch(${name}, ivec2(x, y), 0)${vec4 ? '' : '.r'};
}
`;
  } else {
    return `
uniform ${samplerPrefix}sampler2DArray ${name};
${uniforms}

${ioType} get_${name}(${args}) {
int flat_index = ${flat_index};
ivec3 texture_wh = textureSize(${name}, 0);
int texture_w = texture_wh.x;
int texture_h = texture_wh.y;
int y = flat_index / texture_w;
int x = flat_index - y * texture_w;
int z = y / texture_h;
y = y - z * texture_h;
return texelFetch(${name}, ivec3(x, y, z), 0)${vec4 ? '' : '.r'};
}
`;
  }
}

export function shaderGenTensorNDGet(
  name: string,
  ndim: number,
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault
): string {
  return shaderGenTensorNDGetInternal(name, ndim, dim, dtype, false);
}

export function shaderGenTensorNDGetVec4(
  name: string,
  ndim: number,
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault
): string {
  return shaderGenTensorNDGetInternal(name, ndim, dim, dtype, true);
}

export function shaderGenTensorNDGetUniformItem(
  name: string,
  tensor: WebGLTensor,
  customStrides?: ReadonlyArray<number>
): WebGLUniformItem[] {
  const strides = customStrides || tensor.strides;
  const uniforms: WebGLUniformItem[] = [];
  for (let i = 0; i < strides.length; i++) {
    uniforms.push({
      name: `_ka_${name}_stride_${i}`,
      type: 'int',
      value: strides[i],
    });
  }
  return uniforms;
}

function shaderGenTensorOutputUniformItemInternal(
  tensor: WebGLTensor,
  customShape?: ReadonlyArray<number>,
  elementwise?: boolean
): WebGLUniformItem[] {
  const shape = customShape || tensor.shape;
  const name = 'tex_output',
    uniforms: WebGLUniformItem[] = [];
  if (!elementwise) {
    for (let i = 0; i < shape.length; i++) {
      uniforms.push({
        name: `_ka_${name}_shape_${i}`,
        type: 'int',
        value: shape[i],
      });
    }
  }
  uniforms.push({
    name: `_ka_${name}_texture_h`,
    type: 'int',
    value: tensor.buffer.textureShape.height,
  });
  uniforms.push({
    name: `_ka_${name}_texture_w`,
    type: 'int',
    value: tensor.buffer.textureShape.width,
  });
  return uniforms;
}

export function shaderGenTensorOutputUniformItem(
  tensor: WebGLTensor,
  customShape?: ReadonlyArray<number>
): WebGLUniformItem[] {
  return shaderGenTensorOutputUniformItemInternal(tensor, customShape);
}

export function shaderGenTensorOutputUniformElementwiseItem(
  tensor: WebGLTensor
): WebGLUniformItem[] {
  return shaderGenTensorOutputUniformItemInternal(tensor, undefined, true);
}

export function shaderGenTensorOutputUniformInternal(
  ndim: number,
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault,
  elementwise?: boolean
): string {
  let source = `
uniform int _ka_tex_output_texture_h;
uniform int _ka_tex_output_texture_w;
`;
  if (dim === '2DArray') {
    source += 'uniform int _ka_depth;'; // Uniformの値はWebGLContext.runKernel内で設定される
  }
  const { vec4Type } = getTypeForDType(dtype, false);
  source += `out ${vec4Type} fragColor;\n`;
  if (!elementwise) {
    for (let i = 0; i < ndim; i++) {
      source += `uniform int _ka_tex_output_shape_${i};`;
    }
  }
  return source;
}

export function shaderGenTensorOutputUniform(
  ndim: number,
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault
): string {
  return shaderGenTensorOutputUniformInternal(ndim, dim, dtype, false);
}

export function shaderGenTensorOutputUniformElementwise(
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault
): string {
  return shaderGenTensorOutputUniformInternal(1, dim, dtype, true);
}

export function shaderGenTensorOutputCoordsWithReturn(
  ndim: number,
  dim: TensorTextureShapeDim
): string {
  let source: string;
  switch (ndim) {
    case 0:
      source = `
int tex_output_0 = 0;
if (tex_output_0 >= 1) {
  return;
}
`;
      break;
    case 1:
      source = `
int tex_output_0 = tex_output_flat;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
`;
      break;
    case 2:
      source = `
int _ka_tmp1 = tex_output_flat / _ka_tex_output_shape_1;
int tex_output_1 = tex_output_flat - _ka_tmp1 * _ka_tex_output_shape_1;
int tex_output_0 = _ka_tmp1;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
`;
      break;
    case 3:
      source = `
int _ka_tmp2 = tex_output_flat / _ka_tex_output_shape_2;
int tex_output_2 = tex_output_flat - _ka_tmp2 * _ka_tex_output_shape_2;
int _ka_tmp1 = _ka_tmp2 / _ka_tex_output_shape_1;
int tex_output_1 = _ka_tmp2 - _ka_tmp1 * _ka_tex_output_shape_1;
int tex_output_0 = _ka_tmp1;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
`;
      break;
    case 4:
      source = `
int _ka_tmp3 = tex_output_flat / _ka_tex_output_shape_3;
int tex_output_3 = tex_output_flat - _ka_tmp3 * _ka_tex_output_shape_3;
int _ka_tmp2 = _ka_tmp3 / _ka_tex_output_shape_2;
int tex_output_2 = _ka_tmp3 - _ka_tmp2 * _ka_tex_output_shape_2;
int _ka_tmp1 = _ka_tmp2 / _ka_tex_output_shape_1;
int tex_output_1 = _ka_tmp2 - _ka_tmp1 * _ka_tex_output_shape_1;
int tex_output_0 = _ka_tmp1;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
`;
      break;
    case 5:
      source = `
int _ka_tmp4 = tex_output_flat / _ka_tex_output_shape_4;
int tex_output_4 = tex_output_flat - _ka_tmp4 * _ka_tex_output_shape_4;
int _ka_tmp3 = _ka_tmp4 / _ka_tex_output_shape_3;
int tex_output_3 = _ka_tmp4 - _ka_tmp3 * _ka_tex_output_shape_3;
int _ka_tmp2 = _ka_tmp3 / _ka_tex_output_shape_2;
int tex_output_2 = _ka_tmp3 - _ka_tmp2 * _ka_tex_output_shape_2;
int _ka_tmp1 = _ka_tmp2 / _ka_tex_output_shape_1;
int tex_output_1 = _ka_tmp2 - _ka_tmp1 * _ka_tex_output_shape_1;
int tex_output_0 = _ka_tmp1;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
    `;
      break;
    case 6:
      source = `
int _ka_tmp5 = tex_output_flat / _ka_tex_output_shape_5;
int tex_output_5 = tex_output_flat - _ka_tmp5 * _ka_tex_output_shape_5;
int _ka_tmp4 = _ka_tmp5 / _ka_tex_output_shape_4;
int tex_output_4 = _ka_tmp5 - _ka_tmp4 * _ka_tex_output_shape_4;
int _ka_tmp3 = _ka_tmp4 / _ka_tex_output_shape_3;
int tex_output_3 = _ka_tmp4 - _ka_tmp3 * _ka_tex_output_shape_3;
int _ka_tmp2 = _ka_tmp3 / _ka_tex_output_shape_2;
int tex_output_2 = _ka_tmp3 - _ka_tmp2 * _ka_tex_output_shape_2;
int _ka_tmp1 = _ka_tmp2 / _ka_tex_output_shape_1;
int tex_output_1 = _ka_tmp2 - _ka_tmp1 * _ka_tex_output_shape_1;
int tex_output_0 = _ka_tmp1;
if (tex_output_0 >= _ka_tex_output_shape_0) {
  return;
}
      `;
      break;
    default:
      throw new Error();
  }

  /*
   * Gl_FragCoord.x 's precision is mediump, which only has 10bit precision
   * force casting to highp is needed in iOS. Also, "-0.5" cannot be removed.
   */
  if (dim === '2D') {
    return `
highp float _ka_helper_gfcx = gl_FragCoord.x;
highp float _ka_helper_gfcy = gl_FragCoord.y;
int tex_output_flat = int(_ka_helper_gfcx - 0.5) + _ka_tex_output_texture_w * int(_ka_helper_gfcy - 0.5);
${source}
    `;
  } else {
    return `
highp float _ka_helper_gfcx = gl_FragCoord.x;
highp float _ka_helper_gfcy = gl_FragCoord.y;
int tex_output_flat = int(_ka_helper_gfcx - 0.5) + _ka_tex_output_texture_w * (int(_ka_helper_gfcy - 0.5) + _ka_depth * _ka_tex_output_texture_h);
${source}
    `;
  }
}

// shaderGenTensorElementwiseGetUniformは不要
export function shaderGenTensorElementwiseGet(
  name: string,
  dim: TensorTextureShapeDim,
  dtype: DType = DTypeDefault,
  vec4?: boolean
): string {
  const { ioType, samplerPrefix } = getTypeForDType(dtype, vec4);
  // can only be used when input / output has same texture dim, shape
  if (dim === '2D') {
    return `
uniform ${samplerPrefix}sampler2D ${name};

${ioType} get_${name}() {
  return texelFetch(${name}, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0).r;
}
`;
  } else {
    return `
uniform ${samplerPrefix}sampler2DArray ${name};

${ioType} get_${name}() {
  return texelFetch(${name}, ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), _ka_depth), 0).r;
}
`;
  }
}
