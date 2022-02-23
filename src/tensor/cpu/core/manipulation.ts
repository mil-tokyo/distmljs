import { TypedArrayTypes } from '../../../dtype';
import { CPUTensor } from '../cpuTensor';

function repeat_sub(
  dx: TypedArrayTypes,
  xShape: ReadonlyArray<number>,
  xStrides: ReadonlyArray<number>,
  repeats: ReadonlyArray<number>,
  axis: number
): Array<any> {
  let dy: Array<any> = [];
  if(axis==0){
    for(let i=0; i<xShape[0]; ++i){
      for(let j=0; j<repeats[i]; ++j){
        dy = [...dy, ...dx.slice(i*xStrides[0], (i+1)*xStrides[0])];
      }
    }
  }else{
    for(let i=0; i<xShape[0]; ++i){
      dy = [...dy, ...repeat_sub(dx.slice(i*xStrides[0], (i+1)*xStrides[0]), xShape.slice(1), xStrides.slice(1), repeats, axis-1)];
    }
  }
  return dy;
}

export function repeat(
  x: CPUTensor,
  repeats: ReadonlyArray<number> | number,
  axis?: number
): CPUTensor {
  if(axis==undefined){  //軸指定がないとき
    if(x.ndim==0){  //xがスカラーのとき
      const y = CPUTensor.zeros([repeats as number], x.dtype);
      const dx = x.getBuffer().data;
      const dy = y.getBuffer().data;
      for(let i=0; i<(repeats as number); ++i){
        dy[i] = dx[0];
      }
      return y;
    }else{  //xが多次元配列のとき
      const y = CPUTensor.zeros([x.getBuffer().length*(repeats as number)], x.dtype);
      const dx = x.getBuffer().data;
      const dy = y.getBuffer().data;
      for(let i=0; i<x.getBuffer().length; ++i){
        for(let j=0; j<(repeats as number); ++j){
          dy[i*(repeats as number)+j] = dx[i];
        }
      }
      return y;
    }
  }else{  //軸指定があるとき
    if(typeof repeats === 'number'){ //repeatsがnumber型のとき
      const y_shape: number[] = [];
      for(let i=0; i<x.shape.length; ++i){
        if(i==axis){
          y_shape[i] = x.shape[i]*(repeats as number);
        }else{
          y_shape[i] = x.shape[i];
        }
      }
      const dx = x.getBuffer().data;
      let new_repeats: number[];
      new_repeats = [];
      for(let i=0; i<x.shape[axis]; ++i){
        new_repeats = [...new_repeats, (repeats as number)];
      }
      const dy = repeat_sub(dx, x.shape, x.strides, new_repeats, axis);
      const y = CPUTensor.fromArray(dy, y_shape, x.dtype);
      return y;
    }else{ //repeatsが配列の時
      const y_shape: number[] = [];
      for(let i=0; i<x.shape.length; ++i){
        if(i==axis){
          let sum = 0;
          for(const e of (repeats as ReadonlyArray<number>)){
            sum += e;
          }
          y_shape[i] = sum;
        }else{
          y_shape[i] = x.shape[i];
        }
      }
      const dx = x.getBuffer().data;
      const dy = repeat_sub(dx, x.shape, x.strides, (repeats as ReadonlyArray<number>), axis);
      const y = CPUTensor.fromArray(dy, y_shape, x.dtype);
      return y;
    }
  }
  throw new Error();
}

export function tile(
  x: CPUTensor,
  reps: ReadonlyArray<number> | number
): CPUTensor {
  // TODO: implement
  throw new Error();
}
