import { assert } from 'chai';
import { CPUTensor } from '../../tensor/cpu/cpuTensor';
import {
  TensorDeserializer,
  TensorSerializer,
} from '../../tensor/serializer/tensorSerializer';
import { base64ToUint8Array, nonNull } from '../../util';
import { arrayNearlyEqual } from '../testUtil';

describe('serializer', () => {
  it('deserialize tensor', () => {
    // created by python code
    const base64 =
      'V0ROMlRFTlMwAAAAABgAAAABAgIAAAADAAAAAQAAAHgAAAAAC38MP6YWNz+zTho/d30LP0jp2D5RWSU/VEVOU00AAAAAMAAAAAYDAwAAAAEAAAAEAAAAAgAAAHl5AAAAAOR09v8X+Pv/Bvr7/5h79f8YWAMATIv9//r2+f+BAPn/pjv//+cH+/8XZvb/GBz8/1RFTlMTAAAAAAEAAAACAAMAAAB6enoAAAAAUVRFTlMZAAAAAAIAAAAJAQIAAAAEAAAAd3d3dwAAAAABAENMT1MAAAAA';
    const d = new TensorDeserializer();
    const tensors = d.deserialize(base64ToUint8Array(base64));
    const x = nonNull(tensors.get('x'));
    assert.equal(x.dtype, 'float32');
    assert.deepEqual(x.shape, [2, 3]);
    arrayNearlyEqual(
      x.toArray(),
      [
        0.54881352186203, 0.7151893377304077, 0.6027633547782898,
        0.5448831915855408, 0.42365479469299316, 0.6458941102027893,
      ]
    );
    const yy = nonNull(tensors.get('yy'));
    assert.equal(yy.dtype, 'int32');
    assert.deepEqual(yy.shape, [3, 1, 4]);
    arrayNearlyEqual(
      yy.toArray(),
      [
        -625436, -264169, -263674, -689256, 219160, -160948, -395526, -458623,
        -50266, -325657, -629225, -254952,
      ]
    );
    const zzz = nonNull(tensors.get('zzz'));
    assert.equal(zzz.dtype, 'uint8');
    assert.deepEqual(zzz.shape, []);
    arrayNearlyEqual(zzz.toArray(), [81]);
    const wwww = nonNull(tensors.get('wwww'));
    assert.equal(wwww.dtype, 'bool');
    assert.deepEqual(wwww.shape, [2]);
    arrayNearlyEqual(wwww.toArray(), [1, 0]);
  });

  it('serialize tensor', () => {
    // シリアライズ結果が一意に定まらない(Mapの順序)ため、デシリアライズして結果比較
    const srcMap = new Map<string, CPUTensor>();

    srcMap.set(
      'x',
      CPUTensor.fromArray(
        [
          0.54881352186203, 0.7151893377304077, 0.6027633547782898,
          0.5448831915855408, 0.42365479469299316, 0.6458941102027893,
        ],
        [2, 3],
        'float32'
      )
    );
    srcMap.set(
      'yy',
      CPUTensor.fromArray(
        [
          -625436, -264169, -263674, -689256, 219160, -160948, -395526, -458623,
          -50266, -325657, -629225, -254952,
        ],
        [3, 1, 4],
        'int32'
      )
    );
    srcMap.set('zzz', CPUTensor.fromArray([81], [], 'uint8'));
    srcMap.set('wwww', CPUTensor.fromArray([1, 0], [2], 'bool'));
    const s = new TensorSerializer();
    const serialized = s.serialize(srcMap);
    const d = new TensorDeserializer();
    const tensors = d.deserialize(serialized);
    const x = nonNull(tensors.get('x'));
    assert.equal(x.dtype, 'float32');
    assert.deepEqual(x.shape, [2, 3]);
    arrayNearlyEqual(
      x.toArray(),
      [
        0.54881352186203, 0.7151893377304077, 0.6027633547782898,
        0.5448831915855408, 0.42365479469299316, 0.6458941102027893,
      ]
    );
    const yy = nonNull(tensors.get('yy'));
    assert.equal(yy.dtype, 'int32');
    assert.deepEqual(yy.shape, [3, 1, 4]);
    arrayNearlyEqual(
      yy.toArray(),
      [
        -625436, -264169, -263674, -689256, 219160, -160948, -395526, -458623,
        -50266, -325657, -629225, -254952,
      ]
    );
    const zzz = nonNull(tensors.get('zzz'));
    assert.equal(zzz.dtype, 'uint8');
    assert.deepEqual(zzz.shape, []);
    arrayNearlyEqual(zzz.toArray(), [81]);
    const wwww = nonNull(tensors.get('wwww'));
    assert.equal(wwww.dtype, 'bool');
    assert.deepEqual(wwww.shape, [2]);
    arrayNearlyEqual(wwww.toArray(), [1, 0]);
  });
});
