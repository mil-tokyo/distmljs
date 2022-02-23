import * as K from 'kakiage';
import Variable = K.nn.Variable;
import F = K.nn.functions;
import L = K.nn.layers;

class BasicBlock extends K.nn.core.Layer {
  conv1: L.Conv2d;
  bn1: L.BatchNorm;
  conv2: L.Conv2d;
  bn2: L.BatchNorm;
  downsampleConv?: L.Conv2d;
  downsampleBN?: L.BatchNorm;

  constructor(
    inPlanes: number,
    planes: number,
    stride: number,
    downsample: boolean
  ) {
    super();
    this.conv1 = new L.Conv2d(inPlanes, planes, 3, {
      stride: stride,
      padding: 1,
      bias: false,
    });
    this.bn1 = new L.BatchNorm(planes, {});
    this.conv2 = new L.Conv2d(planes, planes, 3, { padding: 1, bias: false });
    this.bn2 = new L.BatchNorm(planes, {});
    if (downsample) {
      this.downsampleConv = new L.Conv2d(inPlanes, planes, 1, {
        stride: stride,
        bias: false,
      });
      this.downsampleBN = new L.BatchNorm(planes, {});
    }
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    const x = inputs[0];
    let y = x;
    y = await this.conv1.c(y);
    y = await this.bn1.c(y);
    y = await F.relu(y);
    y = await this.conv2.c(y);
    y = await this.bn2.c(y);
    let ds: Variable;
    if (this.downsampleConv && this.downsampleBN) {
      ds = await this.downsampleConv.c(x);
      ds = await this.downsampleBN.c(ds);
    } else {
      ds = x;
    }
    y = await F.add(ds, y);
    y = await F.relu(y);
    return [y];
  }
}

export class ResNet18 extends K.nn.core.Layer {
  conv1: L.Conv2d;
  bn1: L.BatchNorm;
  fc: L.Linear;
  layer1: L.Sequential;
  layer2: L.Sequential;
  layer3: L.Sequential;
  layer4: L.Sequential;

  constructor(nClass: number) {
    super();
    this.conv1 = new L.Conv2d(3, 64, 7, { stride: 2, padding: 3, bias: false });
    this.bn1 = new L.BatchNorm(64, {});
    this.layer1 = this.makeLayer(64, 64, 2, 1);
    this.layer2 = this.makeLayer(64, 128, 2, 2);
    this.layer3 = this.makeLayer(128, 256, 2, 2);
    this.layer4 = this.makeLayer(256, 512, 2, 2);
    this.fc = new L.Linear(512, nClass);
    this.initConvWeights();
  }

  private makeLayer(
    inPlanes: number,
    planes: number,
    blocks: number,
    stride: number
  ): L.Sequential {
    const layers: K.nn.core.Layer[] = [];
    layers.push(new BasicBlock(inPlanes, planes, stride, stride !== 1));
    for (let i = 1; i < blocks; i++) {
      layers.push(new BasicBlock(planes, planes, 1, false));
    }
    return new L.Sequential(...layers);
  }

  private initConvWeights() {
    for (const param of this.parameters()) {
      if (param.data.ndim === 4) {
        this.initConvWeight(param);
      }
    }
  }

  private initConvWeight(param: K.nn.core.Parameter) {
    // special initialization as in PyTorch's sample
    const rnd = K.math.Random.getDefault();
    const s = param.data.shape;
    const fanOut = s[0] * s[2] * s[3];
    const gain = Math.SQRT2;
    const std = gain / Math.sqrt(fanOut);
    const length = param.data.size;
    const vec = rnd.normal(length);
    for (let i = 0; i < length; i++) {
      vec[i] = vec[i] * std;
    }
    (param.data as K.tensor.CPUTensor).setArray(vec);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await this.bn1.c(y);
    y = await F.relu(y);
    y = await F.max_pool2d(y, { kernelSize: 3, stride: 2, padding: 1 });
    y = await this.layer1.c(y);
    y = await this.layer2.c(y);
    y = await this.layer3.c(y);
    y = await this.layer4.c(y);
    y = await F.adaptive_avg_pool2d(y, 1);
    y = await F.flatten(y);
    y = await this.fc.c(y);
    return [y];
  }
}

export class MiniConvNet extends K.nn.core.Layer {
  conv1: L.Conv2d;
  conv2: L.Conv2d;
  conv3: L.Conv2d;
  fc: L.Linear;

  constructor(nClass: number) {
    super();
    this.conv1 = new L.Conv2d(3, 32, 3, { stride: 2, padding: 1 });
    this.conv2 = new L.Conv2d(32, 32, 3, { stride: 2, padding: 1 });
    this.conv3 = new L.Conv2d(32, 32, 3, { stride: 2, padding: 1 });
    this.fc = new L.Linear(32, nClass);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await F.relu(y);
    y = await this.conv2.c(y);
    y = await F.relu(y);
    y = await this.conv3.c(y);
    y = await F.relu(y);
    y = await F.adaptive_avg_pool2d(y, 1);
    y = await F.flatten(y);
    y = await this.fc.c(y);
    return [y];
  }
}

export class MiniConvNetBN extends K.nn.core.Layer {
  conv1: L.Conv2d;
  bn1: L.BatchNorm;
  conv2: L.Conv2d;
  bn2: L.BatchNorm;
  conv3: L.Conv2d;
  bn3: L.BatchNorm;
  fc: L.Linear;

  constructor(nClass: number) {
    super();
    this.conv1 = new L.Conv2d(3, 32, 3, { stride: 2, padding: 1, bias: false });
    this.bn1 = new L.BatchNorm(32, {});
    this.conv2 = new L.Conv2d(32, 32, 3, {
      stride: 2,
      padding: 1,
      bias: false,
    });
    this.bn2 = new L.BatchNorm(32, {});
    this.conv3 = new L.Conv2d(32, 32, 3, {
      stride: 2,
      padding: 1,
      bias: false,
    });
    this.bn3 = new L.BatchNorm(32, {});
    this.fc = new L.Linear(32, nClass);
  }

  async forward(inputs: Variable[]): Promise<Variable[]> {
    let y = inputs[0];
    y = await this.conv1.c(y);
    y = await this.bn1.c(y);
    y = await F.relu(y);
    y = await this.conv2.c(y);
    y = await this.bn2.c(y);
    y = await F.relu(y);
    y = await this.conv3.c(y);
    y = await this.bn3.c(y);
    y = await F.relu(y);
    y = await F.adaptive_avg_pool2d(y, 1);
    y = await F.flatten(y);
    y = await this.fc.c(y);
    return [y];
  }
}
