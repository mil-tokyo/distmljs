import { tidySync } from '../../../tidy';
import { arange, arrayProd } from '../../../util';
import { sum } from '../core';

import {
  assertFloat32R,
  shaderGenOutput,
  shaderGenTensorNDGet,
  shaderGenTensorNDGetUniformItem,
  shaderGenTensorNDGetVec4,
  shaderGenTensorOutputCoordsWithReturn,
  shaderGenTensorOutputUniform,
  shaderGenTensorOutputUniformItem,
  webglShaderHeader,
} from '../core/shaderHelper';
import { getNNWebGLContext } from '../webglContext';
import {
  tensorTextureShapeFormatRGBA16F,
  tensorTextureShapeFormatRGBA32F,
  WebGLTensor,
} from '../webglTensor';

export interface BatchNormParams {
  axis: number;
  training: boolean;
  eps: number;
  momentum?: number;
  trackRunningStats: boolean;
}

export function batch_norm_webgl(
  x: WebGLTensor,
  affine: { weight: WebGLTensor; bias: WebGLTensor } | null,
  runningStats: {
    runningMean: WebGLTensor;
    runningVar: WebGLTensor;
    numBatchesTracked: WebGLTensor;
  } | null,
  params: BatchNormParams
): {
  y: WebGLTensor;
  statsForBackprop: WebGLTensor;
  updatedRunningStats: {
    runningMean: WebGLTensor;
    runningVar: WebGLTensor;
    numBatchesTracked: WebGLTensor;
  } | null;
} {
  assertFloat32R([x], 'batch_norm');
  if (affine) {
    assertFloat32R([affine.weight, affine.bias], 'batch_norm');
  }
  if (runningStats) {
    assertFloat32R(
      [runningStats.runningMean, runningStats.runningVar],
      'batch_norm'
    );
    if (runningStats.numBatchesTracked.dtype !== 'int32') {
      throw new Error(
        `batch_norm: runningStats.numBatchesTracked tensor must be int32`
      );
    }
    if (runningStats.numBatchesTracked.buffer.dimPerPixel !== 1) {
      throw new Error(`batch_norm: dimPerPixel must be 1`);
    }
  }
  const chLength = x.shape[params.axis];
  const chStride = x.strides[params.axis];
  const innerLength = arrayProd(x.shape.slice(params.axis + 1));
  const innerStride = 1;
  const outerLength = arrayProd(x.shape.slice(0, params.axis));
  const outerStride = x.strides[Math.max(0, params.axis - 1)];
  const reduceLength = innerLength * outerLength;
  const ctx = getNNWebGLContext();

  if (chLength > ctx.maxTextureSize) {
    throw new Error('batch_norm: channel length is larger than maxTextureSize');
  }
  const statTextureFormat = ctx.supportsTexture32bit
    ? tensorTextureShapeFormatRGBA32F
    : tensorTextureShapeFormatRGBA16F;
  const stats = WebGLTensor.empty([chLength * 4], 'float32', undefined, {
    dim: '2D',
    width: chLength,
    height: 1,
    ...statTextureFormat,
  });
  const scalings = WebGLTensor.empty([chLength * 4], 'float32', undefined, {
    dim: '2D',
    width: chLength,
    height: 1,
    ...statTextureFormat,
  });

  let updatedRunningStats: {
    runningMean: WebGLTensor;
    runningVar: WebGLTensor;
    numBatchesTracked: WebGLTensor;
  } | null = null;

  if (params.training || !runningStats) {
    const kernelName = `batch_norm_stat_${innerLength}_${outerLength}`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
#define innerLength ${innerLength}
#define outerLength ${outerLength}
float invReduceLength = 1.0 / float(innerLength * outerLength);
  ${shaderGenTensorOutputUniform(1, stats.buffer.textureShape.dim)}
  ${shaderGenTensorNDGet('tex_x', 3, x.buffer.textureShape.dim)}
  void main() {
    ${shaderGenTensorOutputCoordsWithReturn(1, stats.buffer.textureShape.dim)}
    float sum = 0.0, sqsum = 0.0;
    for (int outer = 0; outer < outerLength; outer++) {
      for (int inner = 0; inner < innerLength; inner++) {
        float v = get_tex_x(outer, tex_output_0, inner);
        sum += v;
        sqsum += v * v;
      }
    }
    float mean = sum * invReduceLength;
    float variance = sqsum * invReduceLength - mean * mean;
    vec4 st = vec4(mean, variance, 0.0, 0.0);
    ${shaderGenOutput('st', 'float32', true)};
  }
  `
      );
    }
    ctx.runKernel(kernelName, [{ tensor: x, name: 'tex_x' }], stats, [
      ...shaderGenTensorOutputUniformItem(stats, [chLength]),
      ...shaderGenTensorNDGetUniformItem('tex_x', x, [
        outerStride,
        chStride,
        innerStride,
      ]),
    ]);

    if (params.trackRunningStats) {
      const updatedRunningMean = WebGLTensor.empty([chLength]);
      const updatedRunningVar = WebGLTensor.empty([chLength]);
      const updatedNumBatchesTracked = WebGLTensor.empty([], 'int32');
      updatedRunningStats = {
        runningMean: updatedRunningMean,
        runningVar: updatedRunningVar,
        numBatchesTracked: updatedNumBatchesTracked,
      };
      const momentum = params.momentum;
      if (momentum == null) {
        throw new Error('batch_norm: momentum == null is not yet implemented');
      }

      if (runningStats) {
        // update running stats
        // 計算式については src\tensor\cpu\nnfunction\batch_norm.ts も参照
        {
          const kernelName = `batch_norm_update_running_stats_mean`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        1,
        updatedRunningMean.buffer.textureShape.dim
      )}
      ${shaderGenTensorNDGetVec4('tex_stats', 1, stats.buffer.textureShape.dim)}
      ${shaderGenTensorNDGet(
        'tex_mean',
        1,
        runningStats.runningMean.buffer.textureShape.dim
      )}
      uniform float momentum;
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          1,
          updatedRunningMean.buffer.textureShape.dim
        )}
        vec4 meanvar = get_tex_stats(tex_output_0);
        float m = get_tex_mean(tex_output_0);
        float v = (1.0 - momentum) * m + momentum * meanvar.r;
        ${shaderGenOutput('v')};
      }
      `
            );
          }
          ctx.runKernel(
            kernelName,
            [
              { tensor: stats, name: 'tex_stats' },
              { tensor: runningStats.runningMean, name: 'tex_mean' },
            ],
            updatedRunningMean,
            [
              ...shaderGenTensorOutputUniformItem(updatedRunningMean),
              ...shaderGenTensorNDGetUniformItem('tex_stats', stats),
              ...shaderGenTensorNDGetUniformItem(
                'tex_mean',
                runningStats.runningMean
              ),
              { name: 'momentum', value: momentum, type: 'float' },
            ]
          );
        }

        {
          const kernelName = `batch_norm_update_running_stats_var`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        1,
        updatedRunningVar.buffer.textureShape.dim
      )}
      ${shaderGenTensorNDGetVec4('tex_stats', 1, stats.buffer.textureShape.dim)}
      ${shaderGenTensorNDGet(
        'tex_var',
        1,
        runningStats.runningVar.buffer.textureShape.dim
      )}
      uniform float momentum;
      uniform float coef;
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          1,
          updatedRunningVar.buffer.textureShape.dim
        )}
        vec4 meanvar = get_tex_stats(tex_output_0);
        float m = get_tex_var(tex_output_0);
        float v = (1.0 - momentum) * m + momentum * meanvar.g * coef;
        ${shaderGenOutput('v')};
      }
      `
            );
          }
          ctx.runKernel(
            kernelName,
            [
              { tensor: stats, name: 'tex_stats' },
              { tensor: runningStats.runningVar, name: 'tex_var' },
            ],
            updatedRunningVar,
            [
              ...shaderGenTensorOutputUniformItem(updatedRunningVar),
              ...shaderGenTensorNDGetUniformItem('tex_stats', stats),
              ...shaderGenTensorNDGetUniformItem(
                'tex_var',
                runningStats.runningVar
              ),
              { name: 'momentum', value: momentum, type: 'float' },
              {
                name: 'coef',
                value: reduceLength / (reduceLength - 1),
                type: 'float',
              },
            ]
          );
        }

        {
          const kernelName = `batch_norm_update_running_stats_nbt`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        0,
        updatedNumBatchesTracked.buffer.textureShape.dim,
        updatedNumBatchesTracked.dtype
      )}
      ${shaderGenTensorNDGet(
        'tex_nbt',
        0,
        runningStats.numBatchesTracked.buffer.textureShape.dim,
        runningStats.numBatchesTracked.dtype
      )}
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          0,
          updatedNumBatchesTracked.buffer.textureShape.dim
        )}
        int m = get_tex_nbt();
        int v = m + 1;
        ${shaderGenOutput('v', updatedNumBatchesTracked.dtype)};
      }
      `
            );
          }
          ctx.runKernel(
            kernelName,
            [{ tensor: runningStats.numBatchesTracked, name: 'tex_nbt' }],
            updatedNumBatchesTracked,
            [
              ...shaderGenTensorOutputUniformItem(updatedNumBatchesTracked),
              ...shaderGenTensorNDGetUniformItem(
                'tex_nbt',
                runningStats.numBatchesTracked
              ),
            ]
          );
        }
      } else {
        // make new running stats
        {
          const kernelName = `batch_norm_create_running_stats_mean`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        1,
        updatedRunningMean.buffer.textureShape.dim
      )}
      ${shaderGenTensorNDGetVec4('tex_stats', 1, stats.buffer.textureShape.dim)}
      uniform float momentum;
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          1,
          updatedRunningMean.buffer.textureShape.dim
        )}
        vec4 meanvar = get_tex_stats(tex_output_0);
        float v = momentum * meanvar.r;
        ${shaderGenOutput('v')};
      }
      `
            );
          }
          ctx.runKernel(
            kernelName,
            [{ tensor: stats, name: 'tex_stats' }],
            updatedRunningMean,
            [
              ...shaderGenTensorOutputUniformItem(updatedRunningMean),
              ...shaderGenTensorNDGetUniformItem('tex_stats', stats),
              { name: 'momentum', value: momentum, type: 'float' },
            ]
          );
        }

        {
          const kernelName = `batch_norm_create_running_stats_var`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        1,
        updatedRunningVar.buffer.textureShape.dim
      )}
      ${shaderGenTensorNDGetVec4('tex_stats', 1, stats.buffer.textureShape.dim)}
      uniform float momentum;
      uniform float coef;
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          1,
          updatedRunningVar.buffer.textureShape.dim
        )}
        vec4 meanvar = get_tex_stats(tex_output_0);
        float v = (1.0 - momentum) + momentum * meanvar.g * coef;
        ${shaderGenOutput('v')};
      }
      `
            );
          }
          ctx.runKernel(
            kernelName,
            [{ tensor: stats, name: 'tex_stats' }],
            updatedRunningVar,
            [
              ...shaderGenTensorOutputUniformItem(updatedRunningVar),
              ...shaderGenTensorNDGetUniformItem('tex_stats', stats),
              { name: 'momentum', value: momentum, type: 'float' },
              {
                name: 'coef',
                value: reduceLength / (reduceLength - 1),
                type: 'float',
              },
            ]
          );
        }

        {
          const kernelName = `batch_norm_create_running_stats_nbt`;
          if (!ctx.hasKernel(kernelName)) {
            ctx.addKernel(
              kernelName,
              webglShaderHeader +
                `
      ${shaderGenTensorOutputUniform(
        0,
        updatedNumBatchesTracked.buffer.textureShape.dim,
        updatedNumBatchesTracked.dtype
      )}
      void main() {
        ${shaderGenTensorOutputCoordsWithReturn(
          0,
          updatedNumBatchesTracked.buffer.textureShape.dim
        )}
        int v = 1;
        ${shaderGenOutput('v', updatedNumBatchesTracked.dtype)};
      }
      `
            );
          }
          ctx.runKernel(kernelName, [], updatedNumBatchesTracked, [
            ...shaderGenTensorOutputUniformItem(updatedNumBatchesTracked),
          ]);
        }
      }
    }
  } else {
    // copy stats from runningStats
    if (!runningStats) {
      throw new Error('batch_norm: training == false && runningStats == false');
    }
    const kernelName = `batch_norm_statcopy`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
  ${shaderGenTensorOutputUniform(1, stats.buffer.textureShape.dim)}
  ${shaderGenTensorNDGet(
    'tex_mean',
    1,
    runningStats.runningMean.buffer.textureShape.dim
  )}
  ${shaderGenTensorNDGet(
    'tex_var',
    1,
    runningStats.runningVar.buffer.textureShape.dim
  )}
  void main() {
    ${shaderGenTensorOutputCoordsWithReturn(1, stats.buffer.textureShape.dim)}
    float mean = get_tex_mean(tex_output_0);
    float variance = get_tex_var(tex_output_0);
    vec4 st = vec4(mean, variance, 0.0, 0.0);
    ${shaderGenOutput('st', 'float32', true)};
  }
  `
      );
    }
    ctx.runKernel(
      kernelName,
      [
        { tensor: runningStats.runningMean, name: 'tex_mean' },
        { tensor: runningStats.runningVar, name: 'tex_var' },
      ],
      stats,
      [
        ...shaderGenTensorOutputUniformItem(stats),
        ...shaderGenTensorNDGetUniformItem(
          'tex_mean',
          runningStats.runningMean
        ),
        ...shaderGenTensorNDGetUniformItem('tex_var', runningStats.runningVar),
      ]
    );
  }

  if (affine) {
    const kernelName = `batch_norm_scaling_with_affine`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
  ${shaderGenTensorOutputUniform(1, scalings.buffer.textureShape.dim)}
  ${shaderGenTensorNDGetVec4('tex_stats', 1, stats.buffer.textureShape.dim)}
  ${shaderGenTensorNDGet(
    'tex_weight',
    1,
    affine.weight.buffer.textureShape.dim
  )}
  ${shaderGenTensorNDGet('tex_bias', 1, affine.bias.buffer.textureShape.dim)}
  uniform float eps;
  void main() {
    ${shaderGenTensorOutputCoordsWithReturn(
      1,
      scalings.buffer.textureShape.dim
    )}
    vec4 meanvar = get_tex_stats(tex_output_0);
    float w = get_tex_weight(tex_output_0);
    float b = get_tex_bias(tex_output_0);
    float invstd = inversesqrt(meanvar.g + eps);
    vec4 st = vec4(meanvar.r, invstd, w * invstd, -meanvar.r * invstd * w + b);
    ${shaderGenOutput('st', 'float32', true)};
  }
  `
      );
    }
    ctx.runKernel(
      kernelName,
      [
        { tensor: stats, name: 'tex_stats' },
        { tensor: affine.weight, name: 'tex_weight' },
        { tensor: affine.bias, name: 'tex_bias' },
      ],
      scalings,
      [
        ...shaderGenTensorOutputUniformItem(scalings, [chLength]),
        ...shaderGenTensorNDGetUniformItem('tex_stats', stats),
        ...shaderGenTensorNDGetUniformItem('tex_weight', affine.weight),
        ...shaderGenTensorNDGetUniformItem('tex_bias', affine.bias),
        { name: 'eps', value: params.eps, type: 'float' },
      ]
    );
  } else {
    throw new Error('batch_norm without affine is not yet implemented');
  }

  const y = WebGLTensor.empty(x.shape);
  {
    const kernelName = `batch_norm_normalize`;
    if (!ctx.hasKernel(kernelName)) {
      ctx.addKernel(
        kernelName,
        webglShaderHeader +
          `
  ${shaderGenTensorOutputUniform(3, y.buffer.textureShape.dim)}
  ${shaderGenTensorNDGetVec4(
    'tex_scalings',
    1,
    scalings.buffer.textureShape.dim
  )}
  ${shaderGenTensorNDGet('tex_x', 3, x.buffer.textureShape.dim)}
  void main() {
    ${shaderGenTensorOutputCoordsWithReturn(3, y.buffer.textureShape.dim)}
    vec4 scalings = get_tex_scalings(tex_output_1);
    float x = get_tex_x(tex_output_0, tex_output_1, tex_output_2);
    float v = x * scalings.b + scalings.a;
    ${shaderGenOutput('v')};
  }
  `
      );
    }
    ctx.runKernel(
      kernelName,
      [
        { tensor: scalings, name: 'tex_scalings' },
        { tensor: x, name: 'tex_x' },
      ],
      y,
      [
        ...shaderGenTensorOutputUniformItem(y, [
          outerLength,
          chLength,
          innerLength,
        ]),
        ...shaderGenTensorNDGetUniformItem('tex_scalings', scalings),
        ...shaderGenTensorNDGetUniformItem('tex_x', x, [
          outerStride,
          chStride,
          innerStride,
        ]),
      ]
    );
  }

  return { y, updatedRunningStats, statsForBackprop: scalings };
}

export function batch_norm_backprop_webgl(
  x: WebGLTensor,
  gy: WebGLTensor,
  statsForBackprop: WebGLTensor,
  axis: number
): {
  gx: WebGLTensor;
  gweight: WebGLTensor;
  gbias: WebGLTensor;
} {
  // TODO: 高速化
  const axesExceptCh = arange(gy.ndim);
  axesExceptCh.splice(axis, 1);

  const ctx = getNNWebGLContext();
  const [gx, gweight, gbias] = tidySync(() => {
    const gbias = sum(gy, axesExceptCh, true);
    const chLength = gy.shape[axis];
    const reshapeShape = Array(x.ndim).fill(1);
    reshapeShape[axis] = chLength; // e.g. [1, chLength, 1, 1] for 2d image

    const mean = WebGLTensor.empty(reshapeShape);
    const invStd = WebGLTensor.empty(reshapeShape);
    const scale = WebGLTensor.empty(reshapeShape);
    {
      const kernelName = `batch_norm_back_extract_0`;
      if (!ctx.hasKernel(kernelName)) {
        ctx.addKernel(
          kernelName,
          webglShaderHeader +
            `
    ${shaderGenTensorOutputUniform(1, mean.buffer.textureShape.dim)}
    ${shaderGenTensorNDGetVec4(
      'tex_scalings',
      1,
      statsForBackprop.buffer.textureShape.dim
    )}
    void main() {
      ${shaderGenTensorOutputCoordsWithReturn(1, mean.buffer.textureShape.dim)}
      vec4 scalings = get_tex_scalings(tex_output_0);
      float v = scalings.r;
      ${shaderGenOutput('v')};
    }
    `
        );
      }
      ctx.runKernel(
        kernelName,
        [{ tensor: statsForBackprop, name: 'tex_scalings' }],
        mean,
        [
          ...shaderGenTensorOutputUniformItem(mean, [chLength]),
          ...shaderGenTensorNDGetUniformItem('tex_scalings', statsForBackprop),
        ]
      );
    }
    {
      const kernelName = `batch_norm_back_extract_1`;
      if (!ctx.hasKernel(kernelName)) {
        ctx.addKernel(
          kernelName,
          webglShaderHeader +
            `
    ${shaderGenTensorOutputUniform(1, invStd.buffer.textureShape.dim)}
    ${shaderGenTensorNDGetVec4(
      'tex_scalings',
      1,
      statsForBackprop.buffer.textureShape.dim
    )}
    void main() {
      ${shaderGenTensorOutputCoordsWithReturn(
        1,
        invStd.buffer.textureShape.dim
      )}
      vec4 scalings = get_tex_scalings(tex_output_0);
      float v = scalings.g;
      ${shaderGenOutput('v')};
    }
    `
        );
      }
      ctx.runKernel(
        kernelName,
        [{ tensor: statsForBackprop, name: 'tex_scalings' }],
        invStd,
        [
          ...shaderGenTensorOutputUniformItem(invStd, [chLength]),
          ...shaderGenTensorNDGetUniformItem('tex_scalings', statsForBackprop),
        ]
      );
    }

    {
      const kernelName = `batch_norm_back_extract_2`;
      if (!ctx.hasKernel(kernelName)) {
        ctx.addKernel(
          kernelName,
          webglShaderHeader +
            `
    ${shaderGenTensorOutputUniform(1, scale.buffer.textureShape.dim)}
    ${shaderGenTensorNDGetVec4(
      'tex_scalings',
      1,
      statsForBackprop.buffer.textureShape.dim
    )}
    void main() {
      ${shaderGenTensorOutputCoordsWithReturn(1, scale.buffer.textureShape.dim)}
      vec4 scalings = get_tex_scalings(tex_output_0);
      float v = scalings.b;
      ${shaderGenOutput('v')};
    }
    `
        );
      }
      ctx.runKernel(
        kernelName,
        [{ tensor: statsForBackprop, name: 'tex_scalings' }],
        scale,
        [
          ...shaderGenTensorOutputUniformItem(scale, [chLength]),
          ...shaderGenTensorNDGetUniformItem('tex_scalings', statsForBackprop),
        ]
      );
    }

    const gweight = WebGLTensor.sub(
      WebGLTensor.mul(sum(WebGLTensor.mul(x, gy), axesExceptCh, true), invStd),
      WebGLTensor.mul(gbias, WebGLTensor.mul(mean, invStd))
    );
    const tmp = WebGLTensor.mul(
      WebGLTensor.add(
        WebGLTensor.mul(
          WebGLTensor.sub(x, mean),
          WebGLTensor.mul(invStd, gweight)
        ),
        gbias
      ),
      WebGLTensor.s(1.0 / (gy.size / gbias.size))
    );
    const gx = WebGLTensor.mul(scale, WebGLTensor.sub(gy, tmp));
    return [gx, gweight.reshape([-1]), gbias.reshape([-1])];
  });

  return { gx, gweight, gbias };
}
