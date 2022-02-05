import { Variable } from './nn';
import { Tensor } from './tensor';
import { existingBuffers, WebGLTensor } from './tensor/webgl/webglTensor';

type TensorKind = Tensor | Variable;
type TidyResult = TensorKind[];
export function tidy<T extends TidyResult>(fn: () => Promise<T>): Promise<T>;
export function tidy<T extends TidyResult>(
  name: string,
  fn: () => Promise<T>
): Promise<T>;
export async function tidy<T extends TidyResult>(
  nameOrFn: string | (() => Promise<T>),
  fn?: () => Promise<T>
): Promise<T> {
  let fn_: () => Promise<T>;
  let name: string | null;
  if (fn) {
    fn_ = fn;
    name = nameOrFn as string;
  } else {
    fn_ = nameOrFn as () => Promise<T>;
    name = null;
  }
  const keepWebGLBuffers = new Set(existingBuffers);
  const returned = await fn_();
  const lastWebGLBuffers = new Set(existingBuffers);
  // lastBuffers - firstBuffers - returned を開放
  for (const r of returned) {
    if (WebGLTensor.isWebGLTensor(r)) {
      keepWebGLBuffers.add(r.buffer);
    } else if (r instanceof Variable) {
      const rd = r.data;
      if (WebGLTensor.isWebGLTensor(rd)) {
        keepWebGLBuffers.add(rd.buffer);
      }
    }
  }
  for (const fb of keepWebGLBuffers) {
    lastWebGLBuffers.delete(fb);
  }
  let ndispose = 0;
  for (const buf of lastWebGLBuffers) {
    buf.dispose();
    ndispose++;
  }
  console.log(
    `tidy ${name}: disposed ${ndispose}, ${JSON.stringify(
      WebGLTensor.getDebugInfo()
    )}`
  );

  return returned;
}
