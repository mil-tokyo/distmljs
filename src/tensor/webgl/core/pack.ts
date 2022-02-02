export function packToFloat32Array(
  src: ArrayLike<number>,
  length: number
): Float32Array {
  const buffer = new Float32Array(length);
  buffer.set(src);
  return buffer;
}

export function packToFloat16Array(
  src: ArrayLike<number>,
  length: number
): Uint16Array {
  const srcLength = src.length;
  let srcUInt32: Uint32Array;
  if (src instanceof Float32Array) {
    srcUInt32 = new Uint32Array(src.buffer, src.byteOffset, srcLength);
  } else {
    const srcFloat32 = new Float32Array(srcLength);
    srcFloat32.set(src);
    srcUInt32 = new Uint32Array(srcFloat32.buffer);
  }

  const buffer = new Uint16Array(length);
  for (let i = 0; i < srcLength; i++) {
    const x = srcUInt32[i];
    const packed =
      ((x >> 16) & 0x8000) |
      ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
      ((x >> 13) & 0x03ff);
    buffer[i] = packed;
  }
  return buffer;
}

export function packToInt32Array(
  src: ArrayLike<number>,
  length: number
): Int32Array {
  const buffer = new Int32Array(length);
  buffer.set(src);
  return buffer;
}

export function unpackFromFloat32Array(
  src: Float32Array,
  length: number
): Float32Array {
  const buffer = new Float32Array(length);
  const srcView = new Float32Array(src.buffer, src.byteOffset, length);
  buffer.set(srcView);
  return buffer;
}

export function unpackFromFloat16Array(
  src: Uint16Array,
  length: number
): Float32Array {
  const buffer = new Float32Array(length);
  const bufferUInt32 = new Uint32Array(buffer.buffer);
  for (let i = 0; i < length; i++) {
    const h = src[i];
    const unpacked =
      ((h & 0x8000) << 16) |
      (((h & 0x7c00) + 0x1c000) << 13) |
      ((h & 0x03ff) << 13);
    bufferUInt32[i] = unpacked;
  }
  return buffer;
}

export function unpackFromInt32Array(
  src: Int32Array,
  length: number
): Int32Array {
  const buffer = new Int32Array(length);
  const srcView = new Int32Array(src.buffer, src.byteOffset, length);
  buffer.set(srcView);
  return buffer;
}
