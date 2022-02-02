import { getMultiBroadcastShape } from '../../shapeUtil';
import { getNNWebGLContext, webglShaderHeader } from '../webglContext';
import { WebGLTensor } from '../webglTensor';

export function add(lhs: WebGLTensor, rhs: WebGLTensor): WebGLTensor {
  if (lhs.dtype !== 'float32' || rhs.dtype !== 'float32') {
    throw new Error();
  }
  if (lhs.buffer.dimPerPixel !== 1 || rhs.buffer.dimPerPixel !== 1) {
    // TODO
    throw new Error();
  }
  // TODO: binaryに一般化
  const { shape, allStrides } = getMultiBroadcastShape([lhs.shape, rhs.shape]);
  const output = WebGLTensor.empty(shape, 'float32');
  const ctx = getNNWebGLContext();
  if (
    output.buffer.textureShape.dim === '2D' &&
    lhs.buffer.textureShape.dim === '2D' &&
    rhs.buffer.textureShape.dim === '2D'
  ) {
    if (shape.length === 1) {
      const kernelName = `add${shape.length}`;
      if (!ctx.hasKernel(kernelName)) {
        ctx.addKernel(
          kernelName,
          webglShaderHeader +
            `
  uniform sampler2D tex_lhs;
  uniform sampler2D tex_rhs;
  uniform int tex_output_texture_w;
  uniform int tex_output_shape_0;
  uniform int lhs_stride_0;
  float get_lhs(int d0) {
    int flat_index = d0 * lhs_stride_0;
    int texture_w = textureSize(tex_lhs, 0).x;
    int y = flat_index / texture_w;
    int x = flat_index - y * texture_w;
    return texelFetch(tex_lhs, ivec2(x, y), 0).r;
  }
  uniform int rhs_stride_0;
  float get_rhs(int d0) {
    int flat_index = d0 * rhs_stride_0;
    int texture_w = textureSize(tex_rhs, 0).x;
    int y = flat_index / texture_w;
    int x = flat_index - y * texture_w;
    return texelFetch(tex_rhs, ivec2(x, y), 0).r;
  }
  void main() {
    highp float helper_gfcx = gl_FragCoord.x;
    highp float helper_gfcy = gl_FragCoord.y;
    int tex_output_flat = int(helper_gfcx - 0.5) + tex_output_texture_w * int(helper_gfcy - 0.5);  
    int tex_output_0 = tex_output_flat;
    if (tex_output_0 >= tex_output_shape_0) {
      return;
    }

    float v_l = get_lhs(tex_output_0);
    float v_r = get_rhs(tex_output_0);
    float v = v_l + v_r;
    fragColor = vec4(v, 0.0, 0.0, 0.0);
  }
  `
        );
      }
      ctx.runKernel(
        kernelName,
        [
          { tensor: lhs, name: 'tex_lhs' },
          { tensor: rhs, name: 'tex_rhs' },
        ],
        output,
        [
          {
            name: 'tex_output_texture_w',
            type: 'int',
            value: output.buffer.textureShape.width,
          },
          { name: 'tex_output_shape_0', type: 'int', value: shape[0] },
          { name: 'lhs_stride_0', type: 'int', value: allStrides[0][0] },
          { name: 'rhs_stride_0', type: 'int', value: allStrides[1][0] },
        ]
      );
    } else if (shape.length === 2) {
      const kernelName = `add${shape.length}`;
      if (!ctx.hasKernel(kernelName)) {
        ctx.addKernel(
          kernelName,
          webglShaderHeader +
            `
  uniform sampler2D tex_lhs;
  uniform sampler2D tex_rhs;
  uniform int tex_output_texture_w;
  uniform int tex_output_shape_0;
  uniform int tex_output_shape_1;
  uniform int lhs_stride_0;
  uniform int lhs_stride_1;
  float get_lhs(int d0, int d1) {
    int flat_index = d0 * lhs_stride_0 + d1 * lhs_stride_1;
    int texture_w = textureSize(tex_lhs, 0).x;
    int y = flat_index / texture_w;
    int x = flat_index - y * texture_w;
    return texelFetch(tex_lhs, ivec2(x, y), 0).r;
  }
  uniform int rhs_stride_0;
  uniform int rhs_stride_1;
  float get_rhs(int d0, int d1) {
    int flat_index = d0 * rhs_stride_0 + d1 * rhs_stride_1;
    int texture_w = textureSize(tex_rhs, 0).x;
    int y = flat_index / texture_w;
    int x = flat_index - y * texture_w;
    return texelFetch(tex_rhs, ivec2(x, y), 0).r;
  }
  void main() {
    highp float helper_gfcx = gl_FragCoord.x;
    highp float helper_gfcy = gl_FragCoord.y;
    int tex_output_flat = int(helper_gfcx - 0.5) + tex_output_texture_w * int(helper_gfcy - 0.5);  
    int tmp1 = tex_output_flat / tex_output_shape_1;
    int tex_output_1 = tex_output_flat - tmp1 * tex_output_shape_1;
    int tex_output_0 = tmp1;
    if (tex_output_0 >= tex_output_shape_0) {
      return;
    }

    float v_l = get_lhs(tex_output_0, tex_output_1);
    float v_r = get_rhs(tex_output_0, tex_output_1);
    float v = v_l + v_r;
    fragColor = vec4(v, 0.0, 0.0, 0.0);
  }
  `
        );
      }
      ctx.runKernel(
        kernelName,
        [
          { tensor: lhs, name: 'tex_lhs' },
          { tensor: rhs, name: 'tex_rhs' },
        ],
        output,
        [
          {
            name: 'tex_output_texture_w',
            type: 'int',
            value: output.buffer.textureShape.width,
          },
          { name: 'tex_output_shape_0', type: 'int', value: shape[0] },
          { name: 'tex_output_shape_1', type: 'int', value: shape[1] },
          { name: 'lhs_stride_0', type: 'int', value: allStrides[0][0] },
          { name: 'lhs_stride_1', type: 'int', value: allStrides[0][1] },
          { name: 'rhs_stride_0', type: 'int', value: allStrides[1][0] },
          { name: 'rhs_stride_1', type: 'int', value: allStrides[1][1] },
        ]
      );
    } else {
      throw new Error(
        `Output dimension ${shape.length} is not yet implemented`
      );
    }
  } else {
    throw new Error('2Darray input/output not yet implemented');
  }
  return output;
}
