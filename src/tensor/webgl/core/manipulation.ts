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
