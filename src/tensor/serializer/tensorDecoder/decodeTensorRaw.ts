import Long from 'long';
import { TypedArrayTypes } from '../../../dtype';
import { TensorSerializerDataType } from '../tensorSerializer';
import { clipLong } from '../tensorSerializerUtil';

export function decodeTensorRaw(
  buf: ArrayBuffer,
  bodyByteOffset: number,
  bodyCompressedLength: number,
  dataType: number,
  numel: number
): TypedArrayTypes {
  let data: TypedArrayTypes;
  switch (dataType) {
    case TensorSerializerDataType.FLOAT:
      data = new Float32Array(numel);
      break;
    case TensorSerializerDataType.BOOL:
    case TensorSerializerDataType.UINT8:
      data = new Uint8Array(numel);
      break;
    case TensorSerializerDataType.INT32:
      data = new Int32Array(numel);
      break;
    case TensorSerializerDataType.INT64: {
      data = new Int32Array(numel);
      const view = new DataView(buf, bodyByteOffset, numel * 8);
      for (let idx = 0; idx < numel; idx++) {
        data[idx] = clipLong(
          new Long(
            view.getUint32(idx * 8, true),
            view.getUint32(idx * 8 + 4, true)
          )
        );
      }
      return data;
    }
    default:
      throw new Error('Unsupported DataType');
  }
  // Buf may not be aligned
  const dataUint8View = new Uint8Array(data.buffer),
    srcUint8View = new Uint8Array(buf, bodyByteOffset, data.byteLength);
  dataUint8View.set(srcUint8View);
  return data;
}
