import { WebGPUMetaBuffer } from './webgpuMetaBuffer';
import { WebGPUTensor } from './webgpuTensor';

interface WebGPURunnerPipeline {
  bindGroupLayout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}

type WorkGroupDim = 'x' | 'y' | 'z';

export interface WebGPUMetaBufferContentElement {
  value: number;
  type: 'int32' | 'uint32' | 'float32';
}

export interface WebGPUMetaBufferContent {
  elements: WebGPUMetaBufferContentElement[];
}

export interface WebGPURunnerRequest {
  pipelineName: string;
  tensors: WebGPUTensor[];
  meta: WebGPUMetaBufferContent | null;
  workGroups: { [key in WorkGroupDim]: number };
}

export class NNWebGPUContext {
  initialized: boolean;

  isSupported: boolean;

  device!: GPUDevice;

  private pipelines: Map<string, WebGPURunnerPipeline>;

  pooledMetaBuffer: WebGPUMetaBuffer[] = [];

  constructor() {
    if (
      typeof navigator.gpu !== 'object' ||
      typeof navigator.gpu.requestAdapter !== 'function'
    ) {
      throw new Error('WebGPU is not supported on this browser');
    }
    this.initialized = false;
    this.isSupported = false;
    this.pipelines = new Map();
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const adapter = await navigator.gpu!.requestAdapter();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    this.device = (await adapter!.requestDevice()) as GPUDevice;
    if (!this.device) {
      throw new Error('GPUAdapter.requestDevice() returned null');
    }
    this.isSupported = true;
    this.initialized = true;
  }

  hasPipeline(name: string): boolean {
    return this.pipelines.has(name);
  }

  createPipeline(name: string, shader: Uint32Array, nBuffers: number): void {
    if (this.hasPipeline(name)) {
      return;
    }
    const { device } = this,
      bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < nBuffers; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      });
    }
    const bindGroupLayout = device.createBindGroupLayout({
        entries: bindings,
      }),
      pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      shaderModule = device.createShaderModule({ code: shader }),
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          // computeStage?
          module: shaderModule,
          entryPoint: 'main',
        },
      });

    this.pipelines.set(name, { bindGroupLayout, pipeline });
  }

  runKernel(request: WebGPURunnerRequest): void {
    const pipeline = this.pipelines.get(request.pipelineName);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipeline} not found`);
    }
    const { device } = this,
      entries: GPUBindGroupEntry[] = request.tensors.map((t, i) => ({
        binding: i,
        resource: {
          buffer: t.buffer.gpuBuffer,
          size: t.buffer.bufferShape.byteLength,
        },
      }));
    let meta: WebGPUMetaBuffer | null = null;
    if (request.meta) {
      meta = WebGPUMetaBuffer.createBuffer(request.meta);
      entries.push({
        binding: entries.length,
        resource: {
          buffer: meta.buffer.gpuBuffer,
          size: meta.buffer.bufferShape.byteLength,
        },
      });
    }
    const bindGroup = device.createBindGroup({
        layout: pipeline.bindGroupLayout,
        entries,
      }),
      commandEncoder = device.createCommandEncoder(),
      passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(pipeline.pipeline);
    passEncoder.dispatch(
      request.workGroups.x,
      request.workGroups.y,
      request.workGroups.z
    );
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    meta?.pushToPool();
  }
}

let context: NNWebGPUContext | null = null;
export async function initializeNNWebGPUContext(): Promise<void> {
  // 現状非同期処理はないが、将来的に機能テストなどを加える可能性がある
  context = new NNWebGPUContext();
  try {
    await context.initialize();
  } catch (error) {
    context = null;
    throw error;
  }
}

export function getNNWebGPUContext(): NNWebGPUContext {
  if (!context) {
    throw new Error('WebGPU Context does not exist');
  }
  return context;
}
