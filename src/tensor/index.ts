export { Tensor } from './tensor';
export { CPUTensor } from './cpu/cpuTensor';
export { WebGLTensor } from './webgl/webglTensor';
export { initializeNNWebGLContext } from './webgl/webglContext';
export { initializeNNWebGPUContext } from './webgpu/webgpuContext';
export {
  TensorSerializer,
  TensorDeserializer,
} from './serializer/tensorSerializer';
export { slice, Slice, Ellipsis, ellipsis, newaxis } from './slice';
