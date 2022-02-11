import { maxPool2DCalcShape } from '../../nnFunctionUtil';
import {
  assertFloat32R,
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from '../core/shaderHelper';
import { getNNWebGLContext } from '../webglContext';
import { WebGLTensor } from '../webglTensor';

export function max_pool2d_webgl(
  x: WebGLTensor,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    dilation: number | number[];
    returnIndices: false;
    ceilMode: boolean;
  }
): WebGLTensor {
  assertFloat32R([x], 'max_pool2d');
  const {
    batch,
    dilations: [dilations0, dilations1],
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0, pads1],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = maxPool2DCalcShape(params, x.shape);
  const output = WebGLTensor.empty([batch, ch, outShape0, outShape1]);
  const ctx = getNNWebGLContext();
  const kernelName = `max_pool2d_${kernelShape0}_${kernelShape1}`;
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define KS0 ${kernelShape0}
#define KS1 ${kernelShape1}
uniform int ST0;
uniform int ST1;
uniform int DI0;
uniform int DI1;
uniform int PA0;
uniform int PA1;
uniform int IS0;
uniform int IS1;
${shaderGenTensorOutputUniform(4, output.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_x', 4, x.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(4, output.buffer.textureShape.dim)}
  float v = -10000.0;
  for (int k0 = 0; k0 < KS0; k0++) {
    for (int k1 = 0; k1 < KS1; k1++) {
      int in0 = tex_output_2 * ST0 - PA0 + k0 * DI0;
      int in1 = tex_output_3 * ST1 - PA1 + k1 * DI1;
      if (in0 >= 0 && in0 < IS0 && in1 >= 0 && in1 < IS1) {
        float iv = get_tex_x(tex_output_0, tex_output_1, in0, in1);
        if (iv > v) {
          v = iv;
        }
      }
    }
  }
  ${shaderGenOutput('v')};
}
`
    );
  }
  ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_x' }], output, [
    ...shaderGenTensorOutputUniformItem(output),
    ...shaderGenTensorNDGetUniformItem('tex_x', x),
    { name: 'ST0', value: strides0, type: 'int' },
    { name: 'ST1', value: strides1, type: 'int' },
    { name: 'DI0', value: dilations0, type: 'int' },
    { name: 'DI1', value: dilations1, type: 'int' },
    { name: 'PA0', value: pads0, type: 'int' },
    { name: 'PA1', value: pads1, type: 'int' },
    { name: 'IS0', value: inShape0, type: 'int' },
    { name: 'IS1', value: inShape1, type: 'int' },
  ]);
  return output;
}

export function max_pool2d_with_indices_webgl(
  x: WebGLTensor,
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    dilation: number | number[];
    returnIndices: true | 'spatial' | 'flatten';
    ceilMode: boolean;
  }
): WebGLTensor[] {
  // WebGLの制約で、1つのテンソルしか出力できない
  // 最初のカーネルでインデックスを計算し、次にそのインデックスを用いて要素を抽出する

  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
  assertFloat32R([x], 'max_pool2d');
  const {
    batch,
    dilations: [dilations0, dilations1],
    kernelShape: [kernelShape0, kernelShape1],
    pads: [pads0, pads1],
    strides: [strides0, strides1],
    inShape: [inShape0, inShape1],
    outShape: [outShape0, outShape1],
    ch,
  } = maxPool2DCalcShape(params, x.shape);
  const output = WebGLTensor.empty(
    [batch, ch, outShape0, outShape1],
    'float32'
  );
  const outputIdx = WebGLTensor.empty(
    [batch, ch, outShape0, outShape1],
    'int32'
  );
  const ctx = getNNWebGLContext();
  {
    const kernelName = `max_pool2d_find_idx_${kernelShape0}_${kernelShape1}`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
#define KS0 ${kernelShape0}
#define KS1 ${kernelShape1}
uniform int ST0;
uniform int ST1;
uniform int DI0;
uniform int DI1;
uniform int PA0;
uniform int PA1;
uniform int IS0;
uniform int IS1;
${shaderGenTensorOutputUniform(
  4,
  outputIdx.buffer.textureShape.dim,
  outputIdx.dtype
)}
${shaderGenTensorNDGet('tex_x', 4, x.buffer.textureShape.dim)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(4, outputIdx.buffer.textureShape.dim)}
  float v = -10000.0;
  int vidx = 0;
  for (int k0 = 0; k0 < KS0; k0++) {
    for (int k1 = 0; k1 < KS1; k1++) {
      int in0 = tex_output_2 * ST0 - PA0 + k0 * DI0;
      int in1 = tex_output_3 * ST1 - PA1 + k1 * DI1;
      if (in0 >= 0 && in0 < IS0 && in1 >= 0 && in1 < IS1) {
        float iv = get_tex_x(tex_output_0, tex_output_1, in0, in1);
        if (iv > v) {
          v = iv;
          vidx = in0 * IS1 + in1;
        }
      }
    }
  }
  ${shaderGenOutput('vidx', outputIdx.dtype)};
}
`
      );
    }
    ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_x' }], outputIdx, [
      ...shaderGenTensorOutputUniformItem(outputIdx),
      ...shaderGenTensorNDGetUniformItem('tex_x', x),
      { name: 'ST0', value: strides0, type: 'int' },
      { name: 'ST1', value: strides1, type: 'int' },
      { name: 'DI0', value: dilations0, type: 'int' },
      { name: 'DI1', value: dilations1, type: 'int' },
      { name: 'PA0', value: pads0, type: 'int' },
      { name: 'PA1', value: pads1, type: 'int' },
      { name: 'IS0', value: inShape0, type: 'int' },
      { name: 'IS1', value: inShape1, type: 'int' },
    ]);
  }
  {
    // xは3D [batch, ch, spatial] として扱うほうが都合がいい。outputIndexは2D平面をflattenした際のindexであるため。
    const kernelName = `max_pool2d_take_idx`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
${shaderGenTensorOutputUniform(4, output.buffer.textureShape.dim, output.dtype)}
${shaderGenTensorNDGet('tex_x', 3, x.buffer.textureShape.dim)}
${shaderGenTensorNDGet(
  'tex_idx',
  4,
  outputIdx.buffer.textureShape.dim,
  outputIdx.dtype
)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(4, output.buffer.textureShape.dim)}
  int vidx = get_tex_idx(tex_output_0, tex_output_1, tex_output_2, tex_output_3);
  float v = get_tex_x(tex_output_0, tex_output_1, vidx);
  ${shaderGenOutput('v')};
}
`
      );
    }
    ctx.runKernel(
      kernelName,
      [
        { tensor: x, name: 'tex_x' },
        { tensor: outputIdx, name: 'tex_idx' },
      ],
      output,
      [
        ...shaderGenTensorOutputUniformItem(output),
        ...shaderGenTensorNDGetUniformItem('tex_x', x, [
          x.strides[0],
          x.strides[1],
          1,
        ]),
        ...shaderGenTensorNDGetUniformItem('tex_idx', outputIdx),
      ]
    );
  }
  return [output, outputIdx];
}

export function max_pool2d_backprop_webgl(
  indices: WebGLTensor,
  gy: WebGLTensor,
  xShape: ReadonlyArray<number>,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  params: {
    kernelSize: number | number[];
    stride: number | number[];
    padding: number | number[];
    dilation: number | number[];
    ceilMode: boolean;
    returnIndices: true | 'spatial' | 'flatten';
  }
): WebGLTensor {
  if (params.returnIndices === 'flatten') {
    throw new Error('returnIndices==flatten is not yet impelemented');
  }
  assertFloat32R([gy], 'max_pool2d_backprop');
  const [batch, ch, inShape0, inShape1] = xShape;
  const inSpLen = inShape0 * inShape1;
  const [, , outShape0, outShape1] = indices.shape;
  const outSpLen = outShape0 * outShape1;
  const ctx = getNNWebGLContext();
  const gx = WebGLTensor.empty(xShape);
  const kernelName = `max_pool2d_backprop_${outSpLen}`;
  // インデックスは2D平面をflattenした状態の値のため、batch,ch,spatialの3Dで扱う
  // 計算量的には非効率。kernelSizeから探索範囲を絞ったほうが効率的。
  if (!ctx.hasKernel(kernelName)) {
    ctx.addKernel(
      kernelName,
      webglShaderHeader +
        `
#define OSL ${outSpLen}
${shaderGenTensorOutputUniform(3, gx.buffer.textureShape.dim)}
${shaderGenTensorNDGet('tex_gy', 3, gy.buffer.textureShape.dim)}
${shaderGenTensorNDGet(
  'tex_idx',
  3,
  indices.buffer.textureShape.dim,
  indices.dtype
)}
void main() {
  ${shaderGenTensorOutputCoordsWithReturn(3, gx.buffer.textureShape.dim)}
  float v = 0.0;
  for (int i = 0; i < OSL; i++) {
    int idx = get_tex_idx(tex_output_0, tex_output_1, i);
    if (idx == tex_output_2) {
      v += get_tex_gy(tex_output_0, tex_output_1, i);
    }
  }
  ${shaderGenOutput('v')};
}
`
    );
  }
  ctx.runKernel(
    kernelName,
    [
      { tensor: gy, name: 'tex_gy' },
      { tensor: indices, name: 'tex_idx' },
    ],
    gx,
    [
      ...shaderGenTensorOutputUniformItem(gx, [batch, ch, inSpLen]),
      ...shaderGenTensorNDGetUniformItem('tex_gy', gy, [
        gy.strides[0],
        gy.strides[1],
        1,
      ]),
      ...shaderGenTensorNDGetUniformItem('tex_idx', indices, [
        indices.strides[0],
        indices.strides[1],
        1,
      ]),
    ]
  );
  return gx;
}
