import { calcCatShape } from '../../shapeUtil';
import { WebGLUniformItem, getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';
import {
  assertFloat32R,
  makeTensorDimKey,
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from './shaderHelper';

export function cat(tensors: ReadonlyArray<WebGLTensor>, axis = 0): WebGLTensor {
  assertFloat32R(tensors, 'cat');
  const { axisOffsets, yShape, dtype } = calcCatShape(tensors, axis);
  const ndim = yShape.length;

  const output = WebGLTensor.empty(yShape, dtype);
  const ctx = getNNWebGLContext();
  const kernelName = `cat_${dtype}_${tensors.length}_${ndim}_${axis}_${makeTensorDimKey(output, ...tensors)}`;
  if (!ctx.hasKernel(kernelName)) {
    let ndget = '';
    let axis_offset_defs = '';
    for (let i = 0; i < tensors.length; i++) {
      ndget += shaderGenTensorNDGet(`tex_${i}`, ndim, tensors[i].buffer.textureShape.dim, dtype);
      axis_offset_defs += `uniform int axis_offset_${i};`;
    }
    const get_tex_idxs: string[] = [];
    for (let d = 0; d < ndim; d++) {
      if (d === axis) {
        get_tex_idxs.push(`axis_idx`);
      } else {
        get_tex_idxs.push(`tex_output_${d}`);
      }
    }
    const get_tex_idxs_str = get_tex_idxs.join(', ');

    let if_array = '';
    for (let i = tensors.length - 1; i >= 0; i--) {
      if (i === tensors.length - 1) {
        if_array += `if ((axis_idx = tex_output_${axis} - axis_offset_${i}) >= 0) {`;
      } else if (i === 0) {
        if_array += `else { axis_idx = tex_output_${axis};`;
      } else {
        if_array += `else if ((axis_idx = tex_output_${axis} - axis_offset_${i}) >= 0) {`;
      }
      if_array += `v = get_tex_${i}(${get_tex_idxs_str}); }`;
    }
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
      `
${axis_offset_defs}
${shaderGenTensorOutputUniform(ndim, output.buffer.textureShape.dim, dtype)}
${ndget}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(ndim, output.buffer.textureShape.dim)}

  int axis_idx;
  float v;
  ${if_array}
  ${shaderGenOutput('v', dtype)};
}
`
    );
  }

  let uniforms: WebGLUniformItem[] = [];
  uniforms.push(...shaderGenTensorOutputUniformItem(output));
  for (let i = 0; i < tensors.length; i++) {
    uniforms.push(...shaderGenTensorNDGetUniformItem(`tex_${i}`, tensors[i]));
    uniforms.push({ name: `axis_offset_${i}`, value: axisOffsets[i], type: 'int' });
  }
  ctx.runKernel(kernelName, tensors.map((t, i) => ({ tensor: t, name: `tex_${i}` })), output, uniforms);
  return output;
}

export function cat_backprop_webgl(gy: WebGLTensor, shapes: ReadonlyArray<ReadonlyArray<number>>, axis: number): WebGLTensor[] {
  assertFloat32R([gy], 'cat_backprop_webgl');
  const dtype = gy.dtype;
  const axisOffsets: number[] = [];
  let ofs = 0;
  for (let i = 0; i < shapes.length; ++i) {
    axisOffsets.push(ofs);
    ofs += shapes[i][axis];
  }

  const gxs: WebGLTensor[] = [];

  const ctx = getNNWebGLContext();

  for (let i = 0; i < shapes.length; ++i) {
    const axisOffset = axisOffsets[i];
    const gx = WebGLTensor.empty(shapes[i], dtype);
    gxs.push(gx);
    const gxShape = gx.shape;
    const ndim = gxShape.length;

    const kernelName = `catbackprop_${dtype}_${ndim}_${axis}_${makeTensorDimKey(gx, gy)}`;
    if (!ctx.hasKernel(kernelName)) {
      const get_tex_idxs: string[] = [];
      for (let d = 0; d < ndim; d++) {
        if (d === axis) {
          get_tex_idxs.push(`tex_output_${d} + axis_offset`);
        } else {
          get_tex_idxs.push(`tex_output_${d}`);
        }
      }
      const get_tex_idxs_str = get_tex_idxs.join(', ');
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
        `
        uniform int axis_offset;
  ${shaderGenTensorNDGet('tex_input', ndim, gy.buffer.textureShape.dim, dtype)}
  ${shaderGenTensorOutputUniform(ndim, gx.buffer.textureShape.dim, dtype)}
  void main() {
    ${shaderGenTensorOutputCoordsWithReturn(ndim, gx.buffer.textureShape.dim)}
  
    int axis_idx;
    float v = get_tex_input(${get_tex_idxs_str});
    ${shaderGenOutput('v', dtype)};
  }
  `
      );
    }


    ctx.runKernel(kernelName, [{ tensor: gy, name: 'tex_input' }], gx, [
      ...shaderGenTensorOutputUniformItem(gx),
      ...shaderGenTensorNDGetUniformItem('tex_input', gy),
      { name: 'axis_offset', value: axisOffset, type: 'int' },
    ]);
  }
  return gxs;
}
