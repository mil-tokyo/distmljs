from typing import Dict, List, Tuple, Union
import struct
from dataclasses import dataclass
import numpy as np
from .constant_codec_eightbit import compress_tensor_eightbit

FILE_SIGNATURE = b"WDN2"
TENSOR_SIGNATURE = b"TENS"
CLOSE_SIGNATURE = b"CLOS"

DATA_TYPE_TO_NUMPY = {
    1: np.float32,  # onnx.TensorProto.FLOAT
    2: np.uint8,  # onnx.TensorProto.UINT8
    3: np.int8,  # onnx.TensorProto.INT8
    4: np.uint16,  # onnx.TensorProto.UINT16
    5: np.int16,  # onnx.TensorProto.INT16
    6: np.int32,  # onnx.TensorProto.INT32
    7: np.int64,  # onnx.TensorProto.INT64
    9: np.bool,  # onnx.TensorProto.BOOL
    10: np.float16,  # onnx.TensorProto.FLOAT16
    11: np.float64,  # onnx.TensorProto.DOUBLE
    12: np.uint32,  # onnx.TensorProto.UINT32
    13: np.uint64,  # onnx.TensorProto.UINT64
}


def _data_type_from_numpy(np_dtype) -> int:
    # dict like {np.float32: 1} cannot be used due to key equality check
    for k, v in DATA_TYPE_TO_NUMPY.items():
        if v == np_dtype:
            return k
    raise ValueError


def _compress_tensor_raw(data: np.ndarray) -> bytes:
    return data.tobytes()


def _compress_tensor(data: np.ndarray, compression_algorithm: int) -> bytes:
    if compression_algorithm == 0:
        return _compress_tensor_raw(data)
    elif compression_algorithm == 1:
        return compress_tensor_eightbit(data)
    else:
        raise ValueError


def _select_compression_algorithm(data: np.ndarray, compression_algorithm: int) -> int:
    if data.dtype != np.float32:
        return 0
    return compression_algorithm


def _make_tensor_chunk(name: str, data: np.ndarray, compression_algorithm: int) -> bytes:
    data_type = _data_type_from_numpy(data.dtype)
    compression_algorithm = _select_compression_algorithm(
        data, compression_algorithm)
    compressed_body = _compress_tensor(data, compression_algorithm)
    compressed_body_size = len(compressed_body)
    ndim = data.ndim
    dims = data.shape
    name_bytes = name.encode("utf-8")
    name_length = len(name_bytes)
    extra_bytes = b""
    extra_length = len(extra_bytes)
    header = struct.pack("<BIBB",
                         compression_algorithm,
                         compressed_body_size,
                         data_type,
                         ndim)
    header += struct.pack("<" + "I" * len(dims), *dims)
    header += struct.pack("<I", name_length)
    header += name_bytes
    header += struct.pack("<I", extra_length)
    header += extra_bytes
    header += compressed_body
    header = TENSOR_SIGNATURE + struct.pack("<I", len(header)) + header
    return header


def _make_close_chunk() -> bytes:
    return CLOSE_SIGNATURE + b"\0\0\0\0"


def serialize_tensors_to_bytes(tensors: Dict[str, np.ndarray], compression_algorithm: int = 0):
    chunks = [FILE_SIGNATURE]
    for name, data in tensors.items():
        chunks.append(_make_tensor_chunk(name, data, compression_algorithm))
    chunks.append(_make_close_chunk())
    full_data = b"".join(chunks)
    return full_data


def serialize_tensors(path_template: str, tensors: Dict[str, np.ndarray], split_size: int = 0, compression_algorithm: int = 0) -> List[str]:
    full_data = serialize_tensors_to_bytes(tensors, compression_algorithm)
    if split_size <= 0:
        with open(path_template, "wb") as f:
            f.write(full_data)
        return [path_template]
    else:
        file_paths = []
        for i in range((len(full_data) + split_size - 1) // split_size):
            file_path = path_template.format(i)
            file_paths.append(file_path)
            with open(file_path, "wb") as f:
                f.write(full_data[i*split_size:(i+1)*split_size])
        return file_paths


def deserialize_tensor(paths: Union[str, List[str]]) -> Dict[str, np.ndarray]:
    data_all = []
    if isinstance(paths, list):
        p = paths
    else:
        p = [paths]
    for path in p:
        with open(path, "rb") as f:
            data_all.append(f.read())
    return deserialize_tensor_from_bytes(b"".join(data_all))


@dataclass
class ChunkInfo:
    signature: bytes
    body: bytes
    next_offset: int


def deserialize_tensor_from_bytes(data: bytes) -> Dict[str, np.ndarray]:
    if data[:4] != FILE_SIGNATURE:
        raise ValueError("Unexpected file signature")
    offset = 4
    close = False
    tensors = {}
    while not close:
        chunk_info = _extract_chunk(data, offset)
        if chunk_info.signature == TENSOR_SIGNATURE:
            name, array = _parse_tensor_chunk(chunk_info.body)
            tensors[name] = array
        elif chunk_info.signature == CLOSE_SIGNATURE:
            close = True
        offset = chunk_info.next_offset
    return tensors


def _extract_chunk(data: bytes, offset: int) -> ChunkInfo:
    if len(data) < offset + 8:
        raise ValueError("Unexpected EOF")
    signature = data[offset:offset+4]
    body_byte_length = struct.unpack("<I", data[offset+4:offset+8])[0]
    if len(data) < offset + 8 + body_byte_length:
        raise ValueError("Unexpected EOF")
    return ChunkInfo(signature=signature, body=data[offset+8:offset+8+body_byte_length], next_offset=offset+8+body_byte_length)


def _parse_tensor_chunk(body: bytes) -> Tuple[str, np.ndarray]:
    offset = 0
    compression_algorithm, body_compressed_length, data_type, ndim = struct.unpack(
        "<BIBB", body[offset:offset+7])
    offset += 7
    dims = []
    for _ in range(ndim):
        dims.append(struct.unpack("<I", body[offset:offset+4])[0])
        offset += 4
    name_length = struct.unpack("<I", body[offset:offset+4])[0]
    offset += 4
    name = body[offset:offset+name_length].decode("utf-8")
    offset += name_length
    extra_length = struct.unpack("<I", body[offset:offset+4])[0]
    offset += 4
    # extra領域は未使用
    offset += extra_length

    if compression_algorithm != 0:
        raise ValueError("Compression algorithm not implemented")
    numpy_dtype = DATA_TYPE_TO_NUMPY[data_type]
    array = np.frombuffer(
        body[offset:offset+body_compressed_length], dtype=numpy_dtype).reshape(dims)
    return name, array
