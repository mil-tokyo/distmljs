import { WebGPUMetaBufferContent } from './webgpuContext';
import { WebGPUTensorBuffer } from './webgpuTensor';

const metaBufferPool: WebGPUMetaBuffer[] = [];

export class WebGPUMetaBuffer {
  constructor(
    public buffer: WebGPUTensorBuffer,
    private cpuBuffer: Uint32Array,
    private cpuBufferHash: number
  ) {}

  private static buildCPUBuffer(content: WebGPUMetaBufferContent) {
    const cpuBuffer = new Uint32Array(content.elements.length),
      cpuBufferView = new DataView(cpuBuffer.buffer);
    let ofs = 0;
    for (const element of content.elements) {
      switch (element.type) {
        case 'int32':
          cpuBufferView.setInt32(ofs, element.value, true);
          break;
        case 'uint32':
          cpuBufferView.setUint32(ofs, element.value, true);
          break;
        case 'float32':
          cpuBufferView.setFloat32(ofs, element.value, true);
          break;
        default:
          throw new Error();
      }
      ofs += 4;
    }

    return cpuBuffer;
  }

  private static calcBufferHash(cpuBuffer: Uint32Array): number {
    let v = 0;
    for (let i = 0; i < cpuBuffer.length; i++) {
      v += cpuBuffer[i];
    }
    return v;
  }

  private static findPooled(
    cpuBuffer: Uint32Array,
    cpuBufferHash: number
  ): WebGPUMetaBuffer | null {
    const pooled = metaBufferPool;
    for (let i = 0; i < pooled.length; i++) {
      const item = pooled[i];
      if (
        item.cpuBuffer.length === cpuBuffer.length &&
        item.cpuBufferHash === cpuBufferHash
      ) {
        let diff = false;
        for (let j = 0; j < cpuBuffer.length; j++) {
          if (cpuBuffer[j] !== item.cpuBuffer[j]) {
            diff = true;
            break;
          }
        }
        if (!diff) {
          pooled.splice(i, 1);
          return item;
        }
      }
    }
    return null;
  }

  static createBuffer(content: WebGPUMetaBufferContent): WebGPUMetaBuffer {
    const cpuBuffer = WebGPUMetaBuffer.buildCPUBuffer(content),
      cpuBufferHash = WebGPUMetaBuffer.calcBufferHash(cpuBuffer),
      // 全く同じ内容がプールにあればそれを使い、なければバッファ作成とGPUへの転送
      found = WebGPUMetaBuffer.findPooled(cpuBuffer, cpuBufferHash);
    if (found) {
      return found;
    }
    const buf = new WebGPUTensorBuffer(
      {
        byteLength: cpuBuffer.byteLength,
        forWriteFromCPU: true,
        forReadToCPU: false,
      },
      true
    );
    buf.setDataRaw(new Uint32Array(cpuBuffer.buffer));
    return new WebGPUMetaBuffer(buf, cpuBuffer, cpuBufferHash);
  }

  pushToPool(): void {
    metaBufferPool.push(this);
    if (metaBufferPool.length >= 1000) {
      // TODO: remove old buffer
      console.warn('1000 buffers stored to meta buffer pool');
    }
  }
}
