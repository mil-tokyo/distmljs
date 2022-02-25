import { DType, TypedArrayTypes } from '../../dtype';
import {
  arrayProd,
  arraySum,
  base64ToUint8Array,
  uint8ArrayToBase64,
} from '../../util';
import { CPUTensor } from '../cpu/cpuTensor';
import { decodeTensorRaw } from './tensorDecoder/decodeTensorRaw';

const signatureFile = 843990103, // "WDN2"
  signatureTensor = 1397638484, // "TENS"
  signautreClose = 1397705795; // "CLOS"

export enum TensorSerializerDataType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
}

const localStoragePrefix = 'localstorage://';

export class TensorDeserializer {
  async fromHTTP(paths: string[] | string): Promise<Map<string, CPUTensor>> {
    let pathsArray: string[];
    if (typeof paths === 'string') {
      pathsArray = [paths];
    } else {
      pathsArray = paths;
    }
    // TODO: progress
    const fileArray = await this.fetchAllFile(pathsArray);
    return this.deserialize(fileArray);
  }

  async fromLocalStorage(path: string): Promise<Map<string, CPUTensor>> {
    if (!path.startsWith(localStoragePrefix)) {
      throw new Error(
        `toLocalStorage: path must be prefix '${localStoragePrefix}'`
      );
    }
    const key = path.substring(localStoragePrefix.length) + '/0';
    const serialized = localStorage.getItem(key);
    if (serialized) {
      return this.deserialize(base64ToUint8Array(serialized));
    } else {
      throw new Error('no tensor stored');
    }
  }

  /**
   * Load from local file (e.g. files[0] of <input type="file">)
   * @param file
   * @returns
   */
  async fromFile(file: File | Blob): Promise<Map<string, CPUTensor>> {
    const ab = await file.arrayBuffer();
    return this.deserialize(new Uint8Array(ab));
  }

  deserialize(data: Uint8Array): Map<string, CPUTensor> {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    if (signatureFile !== view.getUint32(0, true)) {
      throw new Error('Unexpected file signature');
    }
    let offset = 4;
    const tensors = new Map<string, CPUTensor>();
    let close = false;
    while (!close) {
      const chunkInfo = this.extractChunk(data.buffer, offset);
      switch (chunkInfo.signature) {
        case signatureTensor:
          {
            const { name, tensor } = this.parseTensorChunk(
              data.buffer,
              chunkInfo.bodyByteOffset,
              chunkInfo.bodyByteLength
            );
            tensors.set(name, tensor);
          }
          break;
        case signautreClose:
          close = true;
          break;
      }
      offset = chunkInfo.nextByteOffset;
    }
    return tensors;
  }

  private async fetchAllFile(paths: string[]): Promise<Uint8Array> {
    // TODO: HTTP fetch以外への対応(localStorage等)
    const abs: ArrayBuffer[] = [];
    for (const path of paths) {
      const f = await fetch(path),
        ab = await f.arrayBuffer();
      abs.push(ab);
    }
    const totalLength = arraySum(abs.map((ab) => ab.byteLength)),
      concatArray = new Uint8Array(totalLength);
    let ofs = 0;
    for (const ab of abs) {
      const src = new Uint8Array(ab);
      concatArray.set(src, ofs);
      ofs += src.byteLength;
    }

    return concatArray;
  }

  private extractChunk(
    buf: ArrayBuffer,
    byteOffset: number
  ): {
    signature: number;
    nextByteOffset: number;
    bodyByteOffset: number;
    bodyByteLength: number;
  } {
    const view = new DataView(buf, byteOffset);
    if (view.byteLength < 8) {
      throw new Error('Unexpected EOF');
    }
    const signature = view.getUint32(0, true),
      bodyByteLength = view.getUint32(4, true),
      bodyByteOffset = byteOffset + 8;
    if (view.byteLength < 8 + bodyByteLength) {
      throw new Error('Unexpected EOF');
    }
    const nextByteOffset = bodyByteOffset + bodyByteLength;
    return { signature, bodyByteLength, bodyByteOffset, nextByteOffset };
  }

  private parseTensorChunk(
    buf: ArrayBuffer,
    bodyByteOffset: number,
    bodyByteLength: number
  ): { name: string; tensor: CPUTensor } {
    const view = new DataView(buf, bodyByteOffset, bodyByteLength);

    let ofs = 0;
    const compressionAlgorithm = view.getUint8(ofs);
    ofs += 1;
    const bodyCompressedLength = view.getUint32(ofs, true);
    ofs += 4;
    const dataType = view.getUint8(ofs);
    ofs += 1;
    const ndim = view.getUint8(ofs);
    ofs += 1;
    const dims: number[] = [];
    for (let i = 0; i < ndim; i++) {
      dims.push(view.getUint32(ofs, true));
      ofs += 4;
    }
    const numel = arrayProd(dims),
      nameLength = view.getUint32(ofs, true);
    ofs += 4;
    const name = this.parseString(buf, bodyByteOffset + ofs, nameLength);
    ofs += nameLength;
    const extraLength = view.getUint32(ofs, true);
    ofs += 4;
    // Skip extra data
    ofs += extraLength;

    const data = this.parseTensorBody(
      buf,
      compressionAlgorithm,
      bodyByteOffset + ofs,
      bodyCompressedLength,
      dataType,
      numel
    );
    let dataTypeString: DType;
    switch (dataType) {
      case TensorSerializerDataType.FLOAT:
        dataTypeString = 'float32';
        break;
      case TensorSerializerDataType.INT32:
        dataTypeString = 'int32';
        break;
      case TensorSerializerDataType.BOOL:
        dataTypeString = 'bool';
        break;
      case TensorSerializerDataType.UINT8:
        dataTypeString = 'uint8';
        break;
      default:
        throw new Error('Unsupported DataType');
    }
    const tensor = CPUTensor.fromArray(data, dims, dataTypeString);
    return { name, tensor };
  }

  private parseString(
    buf: ArrayBuffer,
    byteOffset: number,
    byteLength: number
  ): string {
    const view = new Uint8Array(buf, byteOffset, byteLength);
    // TODO: support UTF-8
    return String.fromCharCode(...Array.from(view));
  }

  private parseTensorBody(
    buf: ArrayBuffer,
    compressionAlgorithm: number,
    bodyByteOffset: number,
    bodyCompressedLength: number,
    dataType: number,
    numel: number
  ): TypedArrayTypes {
    switch (compressionAlgorithm) {
      case 0:
        return decodeTensorRaw(
          buf,
          bodyByteOffset,
          bodyCompressedLength,
          dataType,
          numel
        );
      default:
        throw new Error('Unexpected compression algorithm');
    }
  }
}

export class TensorSerializer {
  serialize(
    tensors: Map<string, CPUTensor> | Record<string, CPUTensor> | CPUTensor
  ): Uint8Array {
    let map: Map<string, CPUTensor>;
    if (tensors instanceof Map) {
      map = tensors;
    } else if (tensors instanceof CPUTensor) {
      map = new Map();
      map.set('default', tensors);
    } else {
      map = new Map();
      for (const [k, v] of Object.entries(tensors)) {
        map.set(k, v);
      }
    }
    return this.serializeCore(map);
  }

  async toHTTP(
    tensors: Map<string, CPUTensor> | Record<string, CPUTensor> | CPUTensor,
    path: string
  ): Promise<void> {
    const buf = this.serialize(tensors);
    const blob = new Blob([buf]);
    const f = await fetch(path, { method: 'POST', body: blob });
    if (!f.ok) {
      throw new Error('Server response to save is not ok');
    }
  }

  async toLocalStorage(
    tensors: Map<string, CPUTensor> | Record<string, CPUTensor> | CPUTensor,
    path: string
  ): Promise<void> {
    if (!path.startsWith(localStoragePrefix)) {
      throw new Error(
        `toLocalStorage: path must be prefix '${localStoragePrefix}'`
      );
    }
    const buf = this.serialize(tensors);
    // to support future splitting, save to path/0
    const key = path.substring(localStoragePrefix.length) + '/0';
    localStorage.setItem(key, uint8ArrayToBase64(buf));
  }

  /**
   * Serialize to download file.
   * @param tensors
   * @param fileName
   */
  async toFile(
    tensors: Map<string, CPUTensor> | Record<string, CPUTensor> | CPUTensor,
    fileName = 'tensors.bin'
  ): Promise<void> {
    const buf = this.serialize(tensors);
    const blob = new Blob([buf], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fileName;

    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  private serializeCore(tensors: Map<string, CPUTensor>): Uint8Array {
    let totalLength = 4;
    const tensorEntries: [string, CPUTensor, number, number][] = [];
    for (const [name, tensor] of tensors.entries()) {
      const chunkSize = this.calculateTensorChunkSize(name, tensor);
      tensorEntries.push([name, tensor, totalLength, chunkSize]);
      totalLength += chunkSize;
    }
    const closeChunkOffset = totalLength;
    totalLength += this.calculateCloseChunkSize();
    const dst = new Uint8Array(totalLength);
    const view = new DataView(dst.buffer);
    view.setUint32(0, signatureFile, true);
    for (const [name, tensor, offset, chunkSize] of tensorEntries) {
      this.makeTensorChunk(dst.buffer, offset, chunkSize, name, tensor);
    }
    this.makeCloseChunk(dst.buffer, closeChunkOffset);
    return dst;
  }

  private calculateTensorChunkSize(name: string, tensor: CPUTensor): number {
    // 固定長
    // signature: 4
    // chunk size: 4
    // compressionAlgorithm: 1
    // bodyCompressedLength: 4
    // dataType: 1
    // ndim: 1
    // nameLength: 4
    // extraLength: 4
    // extra: 0
    // 可変長
    // dims: 4 * tensor.ndim
    // name: name.length
    // data: tensor.data.byteLength
    if (!tensor.buffer) {
      throw new Error('buffer for tensor is null');
    }
    return 23 + name.length + tensor.ndim * 4 + tensor.buffer.data.byteLength;
  }

  private makeTensorChunk(
    dst: ArrayBuffer,
    dstOffset: number,
    chunkSize: number,
    name: string,
    tensor: CPUTensor
  ): void {
    const buffer = tensor.buffer;
    if (!buffer) {
      throw new Error('buffer for tensor is null');
    }
    const view = new DataView(dst, dstOffset);
    let ofs = 0;
    view.setUint32(ofs, signatureTensor, true);
    ofs += 4;
    view.setUint32(ofs, chunkSize - 8, true);
    ofs += 4;
    view.setUint8(ofs, 0); // compression algorithm
    ofs += 1;
    view.setUint32(ofs, buffer.data.byteLength, true); // compressed data length
    ofs += 4;
    let dataType: number;
    switch (tensor.dtype) {
      case 'float32':
        dataType = TensorSerializerDataType.FLOAT;
        break;
      case 'int32':
        dataType = TensorSerializerDataType.INT32;
        break;
      case 'uint8':
        dataType = TensorSerializerDataType.UINT8;
        break;
      case 'bool':
        dataType = TensorSerializerDataType.BOOL;
        break;
      default:
        throw new Error('Unsupported dtype');
    }
    view.setUint8(ofs, dataType);
    ofs += 1;
    view.setUint8(ofs, tensor.ndim);
    ofs += 1;
    for (const len of tensor.shape) {
      view.setUint32(ofs, len, true);
      ofs += 4;
    }
    const nameLength = name.length;
    view.setUint32(ofs, nameLength, true);
    ofs += 4;
    for (let i = 0; i < nameLength; i++) {
      view.setUint8(ofs, name.charCodeAt(i));
      ofs += 1;
    }
    view.setUint32(ofs, 0, true); //extraLength
    ofs += 4;
    const dstUint8 = new Uint8Array(dst, dstOffset + ofs);
    const srcUint8 = new Uint8Array(
      buffer.data.buffer,
      buffer.data.byteOffset,
      buffer.data.byteLength
    );
    dstUint8.set(srcUint8);
  }

  private calculateCloseChunkSize(): number {
    return 8;
  }

  private makeCloseChunk(dst: ArrayBuffer, dstOffset: number): void {
    const view = new DataView(dst, dstOffset);
    view.setUint32(0, signautreClose, true);
    view.setUint32(4, 0, true);
  }
}
