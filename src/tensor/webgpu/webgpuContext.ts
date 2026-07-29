import { WebGPUMetaBuffer } from './webgpuMetaBuffer';
import { WebGPUTensor } from './webgpuTensor';

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

  private pipelines: Map<string, GPUComputePipeline>;

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
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('navigator.gpu.requestAdapter() returned null');
    }
    this.device = await adapter.requestDevice();
    void this.device.lost.then((info) => {
      // 一度失われたデバイスは復帰しないため、再初期化が必要となる
      this.initialized = false;
      this.isSupported = false;
      console.error(`WebGPU device is lost (${info.reason}): ${info.message}`);
    });
    this.device.addEventListener('uncapturederror', (event) => {
      console.error(
        'WebGPU uncaptured error:',
        (event as GPUUncapturedErrorEvent).error.message
      );
    });
    this.isSupported = true;
    this.initialized = true;
  }

  hasPipeline(name: string): boolean {
    return this.pipelines.has(name);
  }

  createPipeline(name: string, shader: string): void {
    if (this.hasPipeline(name)) {
      return;
    }
    const { device } = this,
      shaderModule = device.createShaderModule({ code: shader }),
      // バインドグループのレイアウトはWGSLの宣言から自動生成される
      pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

    this.pipelines.set(name, pipeline);
  }

  runKernel(request: WebGPURunnerRequest): void {
    const pipeline = this.pipelines.get(request.pipelineName);
    if (!pipeline) {
      throw new Error(`Pipeline ${request.pipelineName} not found`);
    }
    const { device } = this,
      maxWorkGroups = device.limits.maxComputeWorkgroupsPerDimension;
    for (const dim of ['x', 'y', 'z'] as WorkGroupDim[]) {
      const count = request.workGroups[dim];
      if (count > maxWorkGroups) {
        throw new Error(
          `${request.pipelineName}: workgroup count for the ${dim} dimension ` +
            `(${count}) exceeds the device limit ${maxWorkGroups}`
        );
      }
    }
    const entries: GPUBindGroupEntry[] = request.tensors.map((t, i) => ({
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
        layout: pipeline.getBindGroupLayout(0),
        entries,
      }),
      commandEncoder = device.createCommandEncoder(),
      passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(pipeline);
    passEncoder.dispatchWorkgroups(
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
