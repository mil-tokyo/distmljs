/**
 * Transformer
 * implementation is based on PyTorch
 */

import * as K from 'kakiage';
import Random = K.math.Random;
import CPUTensor = K.tensor.CPUTensor;
import Variable = K.nn.Variable;
import Layer = K.nn.core.Layer;
import Embedding = K.nn.layers.Embedding;
import Sequential = K.nn.layers.Sequential;
import Linear = K.nn.layers.Linear;
import Dropout = K.nn.layers.Dropout;
import LayerNorm = K.nn.layers.LayerNorm;
import add = K.nn.functions.add;
import bmm = K.nn.functions.bmm;
import mul = K.nn.functions.mul;
import relu = K.nn.functions.relu;
import reshape = K.nn.functions.reshape;
import softmax = K.nn.functions.softmax;
import transpose = K.nn.functions.transpose;

export class TransformerEncoder extends Layer {
  layers: Sequential;
  constructor(
    encoderLayerOption: {
      dModel: number;
      nHead: number;
      dimFeedForward?: number;
      dropout?: number;
    },
    numLayers: number
  ) {
    super();

    const layers: Layer[] = [];
    for (let i = 0; i < numLayers; i++) {
      layers.push(
        new TransformerEncoderLayer(
          encoderLayerOption.dModel,
          encoderLayerOption.nHead,
          encoderLayerOption.dimFeedForward,
          encoderLayerOption.dropout
        )
      );
    }
    this.layers = new Sequential(...layers);
  }

  async forward([src, mask]: Variable[]): Promise<Variable[]> {
    let x = src;
    for (let i = 0; i < this.layers.length; i++) {
      x = await this.layers[i].c(x, mask);
    }
    return [x];
  }
}

export class TransformerEncoderLayer extends Layer {
  selfAttn: MultiHeadAttention;
  linear1: Linear;
  linear2: Linear;
  dropout: Dropout;
  norm1: LayerNorm;
  norm2: LayerNorm;
  dropout1: Dropout;
  dropout2: Dropout;

  constructor(
    public readonly dModel: number,
    public readonly nHead: number,
    public readonly dimFeedForward: number = 2048,
    dropout = 0.1
  ) {
    super();
    this.selfAttn = new MultiHeadAttention(dModel, nHead, dropout);
    this.linear1 = new Linear(dModel, dimFeedForward);
    this.dropout = new Dropout(dropout);
    this.linear2 = new Linear(dimFeedForward, dModel);

    this.norm1 = new LayerNorm([dModel], {});
    this.norm2 = new LayerNorm([dModel], {});
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
  }

  async forward([src, srcMask]: Variable[]): Promise<Variable[]> {
    let x = src;
    x = await this.norm1.c(await add(x, await this.saBlock(x, srcMask)));
    x = await this.norm2.c(await add(x, await this.ffBlock(x)));
    return [x];
  }

  private async saBlock(x: Variable, attnMask: Variable): Promise<Variable> {
    const a = await this.selfAttn.c(x, x, x, attnMask);
    return await this.dropout1.c(a);
  }

  private async ffBlock(x: Variable): Promise<Variable> {
    let h = x;
    h = await this.linear1.c(h);
    h = await relu(h);
    h = await this.dropout.c(h);
    h = await this.linear2.c(h);
    h = await this.dropout2.c(h);
    return h;
  }
}

export class MultiHeadAttention extends Layer {
  inProjK: Linear;
  inProjQ: Linear;
  inProjV: Linear;
  outProj: Linear;
  dropout: Dropout;

  constructor(
    public readonly embedDim: number,
    public readonly numHeads: number,
    dropout: number
  ) {
    super();
    this.inProjK = new Linear(embedDim, embedDim);
    this.inProjQ = new Linear(embedDim, embedDim);
    this.inProjV = new Linear(embedDim, embedDim);
    this.outProj = new Linear(embedDim, embedDim);
    this.dropout = new Dropout(dropout);
    this.initWeights();
  }

  private initWeights() {
    // corresponds with pytorch's implementation
    // xavier uniform initialization
    // in PyTorch, inProj* is initialized as one weight of shape [out:embedDim*3, in:embedDim]
    let initRange = Math.sqrt(6 / (this.embedDim * 4)); // fanIn + fanOut: fanIn = embedDim, fanOut = embedDim * 3
    const rnd = Random.getDefault();
    (this.inProjK.weight.data as CPUTensor).setArray(
      rnd.uniform(
        { low: -initRange, high: initRange },
        this.inProjK.weight.data.size
      )
    );
    (this.inProjK.bias!.data as CPUTensor).getBuffer().data.fill(0);
    (this.inProjQ.weight.data as CPUTensor).setArray(
      rnd.uniform(
        { low: -initRange, high: initRange },
        this.inProjQ.weight.data.size
      )
    );
    (this.inProjQ.bias!.data as CPUTensor).getBuffer().data.fill(0);
    (this.inProjV.weight.data as CPUTensor).setArray(
      rnd.uniform(
        { low: -initRange, high: initRange },
        this.inProjV.weight.data.size
      )
    );
    (this.inProjV.bias!.data as CPUTensor).getBuffer().data.fill(0);
    // outProj.weight is default Linear init
    (this.outProj.bias!.data as CPUTensor).getBuffer().data.fill(0);
  }

  async forward([query, key, value, attnMask]: Variable[]): Promise<
    Variable[]
  > {
    const [tgtLen, bsz, embedDim] = query.data.shape;
    const headDim = embedDim / this.numHeads;
    let q = await this.inProjQ.c(query);
    let k = await this.inProjK.c(key);
    let v = await this.inProjV.c(value);
    q = await transpose(
      await reshape(q, [tgtLen, bsz * this.numHeads, headDim]),
      [1, 0, 2]
    );
    k = await transpose(
      await reshape(k, [k.data.shape[0], bsz * this.numHeads, headDim]),
      [1, 2, 0]
    );
    v = await transpose(
      await reshape(v, [v.data.shape[0], bsz * this.numHeads, headDim]),
      [1, 0, 2]
    );

    const [, , E] = q.data.shape;
    q = await mul(q, new Variable(q.data.getClass().s(1 / Math.sqrt(E))));
    let attn = await bmm(q, k);
    attn = await add(attn, attnMask);
    attn = await softmax(attn);
    attn = await this.dropout.c(attn);
    let attnOutput = await bmm(attn, v);
    attnOutput = await transpose(attnOutput, [1, 0, 2]);
    attnOutput = await reshape(attnOutput, [tgtLen * bsz, embedDim]);
    attnOutput = await this.outProj.c(attnOutput);
    attnOutput = await reshape(attnOutput, [tgtLen, bsz, this.embedDim]);

    return [attnOutput];
  }
}

class PositionalEncoding extends K.nn.core.Layer {
  dropout: K.nn.layers.Dropout;
  pe: CPUTensor;
  constructor(public dModel: number, dropout: number, maxLen = 5000) {
    super();
    this.dropout = new K.nn.layers.Dropout(dropout);

    const position = CPUTensor.unsqueeze(
      CPUTensor.fromArray(K.util.arange(maxLen)),
      1
    );
    const divTerm = CPUTensor.exp(
      CPUTensor.mul(
        CPUTensor.fromArray(K.util.arange(0, dModel, 2)),
        CPUTensor.s(-Math.log(10000) / dModel)
      )
    );
    const pe = CPUTensor.zeros([maxLen, 1, dModel]);
    pe.sets(
      CPUTensor.sin(CPUTensor.mul(position, divTerm)),
      K.slice(),
      0,
      K.slice(0, null, 2)
    );
    pe.sets(
      CPUTensor.cos(CPUTensor.mul(position, divTerm)),
      K.slice(),
      0,
      K.slice(1, null, 2)
    );
    this.pe = pe;
  }

  async forward([x]: Variable[]): Promise<Variable[]> {
    let h = x;
    const peSlice = this.pe.gets(K.slice(null, x.data.shape[0]));
    h = await add(h, new Variable(peSlice));
    h = await this.dropout.c(h);
    return [h];
  }
}

export class TransformerModel extends Layer {
  posEncoder: PositionalEncoding;
  transformerEncoder: TransformerEncoder;
  encoder: Embedding;
  decoder: Linear;

  constructor(
    nToken: number,
    public readonly dModel: number,
    nHead: number,
    dHid: number,
    nLayers: number,
    dropout = 0.5
  ) {
    super();
    this.posEncoder = new PositionalEncoding(dModel, dropout);
    this.transformerEncoder = new TransformerEncoder(
      { dModel, nHead, dimFeedForward: dHid, dropout },
      nLayers
    );
    this.encoder = new Embedding(nToken, dModel);
    this.decoder = new Linear(dModel, nToken);
    this.initWeights();
  }

  initWeights() {
    const initRange = 0.1;
    const rnd = Random.getDefault();
    (this.encoder.weight.data as CPUTensor).setArray(
      rnd.uniform(
        { low: -initRange, high: initRange },
        this.encoder.weight.data.size
      )
    );
    (this.decoder.weight.data as CPUTensor).setArray(
      rnd.uniform(
        { low: -initRange, high: initRange },
        this.decoder.weight.data.size
      )
    );
    (this.decoder.bias!.data as CPUTensor).getBuffer().data.fill(0);
  }

  async forward([src, srcMask]: Variable[]): Promise<Variable[]> {
    let x = src;
    x = await this.encoder.c(x);
    x = await mul(x, new Variable(x.data.getClass().s(Math.sqrt(this.dModel))));
    x = await this.posEncoder.c(x);
    let y = await this.transformerEncoder.c(x, srcMask);
    y = await this.decoder.c(y);
    return [y];
  }
}
