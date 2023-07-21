import { throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;
import F = K.nn.functions;
import L = K.nn.layers;
// import { Env } from './cartpole';
import { makeModel } from './models';
import { CPUTensor } from '../../../../dist/tensor';
import { getEnv, Observation } from './Kikyo/exports';

let ws: WebSocket;
let backend: K.Backend

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

const writeState = throttle((message: string) => {
  document.getElementById('state')!.innerText = message;
}, 1000);

const writeBatchInfo = throttle((processedBatches: number, lastLoss: number, batchSize: number) => {
  document.getElementById('table-batches')!.innerText = processedBatches.toString();
  document.getElementById('table-loss')!.innerText = lastLoss.toString();
  document.getElementById('table-batchsize')!.innerText = batchSize.toString();
}, 1000);

const writeTypeInfo = throttle((type: String) => {
  document.getElementById('table-type')!.innerText = type.toString();
}, 1000);

const writeIdInfo = throttle((type: String) => {
  document.getElementById('table-id')!.innerText = type.toString();
}, 1000);

const writeBackendInfo = throttle((type: K.Backend) => {
  document.getElementById('table-backend')!.innerText = String(type).toString();
}, 1000);

async function sendBlob(itemId: string, data: Uint8Array): Promise<void> {
  const blob = new Blob([data]);
  let f = await fetch(`/kakiage/blob/${itemId}`, {
    method: 'PUT',
    body: blob,
    headers: { 'Content-Type': 'application/octet-stream' },
  });
  if (!f.ok) {
    console.log('Error?')
    throw new Error('Server response to save is not ok');
  }
  await f.json();
}

async function recvBlob(itemId: string): Promise<Uint8Array> {
  const f = await fetch(`/kakiage/blob/${itemId}`, { method: 'GET' });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await f.arrayBuffer();
  return new Uint8Array(resp);
}

let model: K.nn.core.Layer;
let model_target: K.nn.core.Layer;

let totalBatches = 0;

async function compute_learner(msg: {
  type: String;
  prioritized: string;
  weight_learner: string;
  weight_target_learner: string;
  dataset: string;
  grad: string;
  td: string;
}) {
  writeTypeInfo(msg.type);
  writeBackendInfo(backend);

  // Load trainable model weights
  const weights_learner = new TensorDeserializer().deserialize(
    await recvBlob(msg.weight_learner)
  );

  for (const { name, parameter } of model.parametersWithName(true, false)) {
    // console.log(name);
    parameter.data = await nonNull(weights_learner.get(name)).to(backend);
    parameter.cleargrad();
  }

  // Load target model weights
  const weights_target_learner = new TensorDeserializer().deserialize(
    await recvBlob(msg.weight_target_learner)
  );
  for (const { name, parameter } of model_target.parametersWithName(true, false)) {
    parameter.data = await nonNull(weights_target_learner.get(name)).to(backend);
    parameter.cleargrad();
  }
  // Load data minibatch
  const dataset = new TensorDeserializer().deserialize(
    await recvBlob(msg.dataset)
  );
  const train_state = new K.nn.Variable(await nonNull(dataset.get('state')).to(backend));
  const action_mask = new K.nn.Variable(await nonNull(dataset.get('action_mask')).to(backend));
  const next_state = new K.nn.Variable(await nonNull(dataset.get('next_state')).to(backend));
  const train_reward = nonNull(dataset.get('reward'));
  const not_done = nonNull(dataset.get('not_done'));
  const discount = nonNull(dataset.get('discount'));
  const batch_size = not_done.size

  // Training
  const target_Q = await model_target.call(next_state);

  let target_Q_max = T.max(T.min(T.cat([T.unsqueeze(await target_Q[0].data.to('cpu'), 0), T.unsqueeze(await target_Q[1].data.to('cpu'), 0)]), 0)[0], 1)[0];

  const label = T.add(train_reward, T.mul(T.mul(not_done, discount), T.unsqueeze(target_Q_max, 1)))
  const labelV = new K.nn.Variable(await label.to(backend))

  const current_Q = await model.call(train_state);
  let cqa, y: K.nn.Variable[], sub, td, loss, lossValue;

  // Currently, only work for DDQN

  cqa = [await F.mul(current_Q[0], action_mask), await F.mul(current_Q[1], action_mask)];
  y = [await F.sum(cqa[0], 1, true), await F.sum(cqa[1], 1, true)];

  sub = [await F.sub(y[0], labelV), await F.sub(y[1], labelV)]
  td = [await F.add(await F.relu(sub[0]), await F.relu(await F.neg(sub[0]))), await F.add(await F.relu(sub[1]), await F.relu(await F.neg(sub[1])))] // abs(sub)

  if (msg.prioritized === "PER" || msg.prioritized === "rand") {
    // conventional MSE // PER
    // loss = [await my_mse(td[0]), await my_mse(td[1])];
    loss = [await K.nn.functions.mseLoss(y[0], labelV), await K.nn.functions.mseLoss(y[1], labelV)];
  } else if (msg.prioritized === "LAP") {
    loss = [await my_huber(td[0]), await my_huber(td[1])];
  } else {
    throw new Error();
  }

  lossValue = (await loss[0].data.to('cpu')).get(0) + (await loss[1].data.to('cpu')).get(0);

  await loss[0].backward();
  await loss[1].backward();

  let tdloss = T.add(await td[0].data.to('cpu'), await td[1].data.to('cpu')).reshape([batch_size]);
  const td_buffer = new Map<string, T>();
  td_buffer.set("td_for_update", tdloss);
  await sendBlob(msg.td, new TensorSerializer().serialize(td_buffer));


  let grads = new Map<string, T>();
  for (const { name, parameter } of model.parametersWithName(true, false)) {
    if (parameter.optimizable) {
      // weight and bias
      grads.set(name, await nonNull(parameter.grad).data.to('cpu'));
    } else {
      // statistics of BN (runningMean, runningVar, numBatchesTracked)
      grads.set(name, await nonNull(parameter.data).to('cpu'));
    }
  }

  await sendBlob(msg.grad, new TensorSerializer().serialize(grads));
  totalBatches += batch_size;
  writeBatchInfo(totalBatches, lossValue, y[0].data.shape[0]);

}


async function my_mse(td: K.nn.Variable): Promise<K.nn.Variable> {
  let batch_size = td.data.shape[0]
  return await F.sum(await F.mul(await F.mul(td, td), new K.nn.Variable(await T.full(td.data.shape, 1 / batch_size).to(backend))));
}

async function my_huber(td: K.nn.Variable): Promise<K.nn.Variable> {
  let batch_size = td.data.shape[0]
  let threshold = new K.nn.Variable(await T.full(td.data.shape, 1).to(backend));

  let above = await F.add(await F.relu(await F.sub(td, threshold)), threshold)
  above = await F.sub(above, new K.nn.Variable(await T.full(td.data.shape, 1).to(backend)));

  let below = await F.add(await F.neg(await F.relu(await F.neg(await F.sub(td, threshold)))), threshold);
  below = await F.mul(await F.mul(below, below), new K.nn.Variable(await T.full(td.data.shape, 1 / 2).to(backend))) // 1以下のTD誤差に対して損失の値が0.5大きくなるけど、逆伝播には関係なさそうだから、まいっか

  return await F.sum(await F.mul(await F.add(above, below), new K.nn.Variable(await T.full(td.data.shape, 1 / batch_size).to(backend))));
}

async function my_weightened_huber(td: K.nn.Variable): Promise<K.nn.Variable> {
  return td
}


async function compute_actor(msg: {
  type: String;
  random_sample: number;
  inputShape: number;
  nClasses: number;
  weight_actor_item_id: string;
  buffer_id: string;
  env_id: string;
}) {

  writeTypeInfo(msg.type + " : " + String(msg.random_sample));
  writeBackendInfo(backend);

  if (msg.random_sample === 0) {

    // Load model weights
    let weight: Uint8Array
    try {
      weight = await recvBlob(msg.weight_actor_item_id);
    } catch (e) {
      return false
    }
    const weights_actor = new TensorDeserializer().deserialize(
      weight
    );
    for (const { name, parameter } of model.parametersWithName(true, false)) {
      parameter.data = await nonNull(weights_actor.get(name)).to(backend);
      parameter.cleargrad();
    }
  }

  // Initialize environment
  let env = await getEnv(msg.env_id);
  let state = T.fromArray((await env.reset()).state);
  let state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
  // let state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));
  const max_episode_len = 200;

  // Initialize buffer
  let buffer_state = T.zeros([max_episode_len, msg.inputShape]);
  let buffer_action = T.zeros([max_episode_len, 1]);
  let buffer_reward = T.zeros([max_episode_len, 1]);
  let buffer_done = T.zeros([max_episode_len, 1]);
  let buffer_td = T.zeros([max_episode_len, 1]);

  // hardcode
  const discount = 0.99

  // Start episode
  for (let step = 0; step < max_episode_len; step++) {

    // Action
    let rnd_sample = Math.random();
    let action: number = 0;
    let y: number[] = [0, 0];

    if (msg.random_sample === 1) {
      let rand_tmp = Math.random();
      action = Math.floor(rand_tmp * msg.nClasses);
    } else {
      let state_input = new K.nn.Variable(await state_norm.reshape([1, msg.inputShape]).to(backend));
      let current_output = await model.call(state_input);
      if (rnd_sample < 0.05) {
        let rand_tmp = Math.random();
        action = Math.floor(rand_tmp * msg.nClasses);
      } else {
        action = T.argmax(await current_output[0].data.to('cpu')).get(0);
      }
      y = [(await current_output[0].data.to('cpu')).reshape([msg.nClasses]).get(action), (await current_output[1].data.to('cpu')).reshape([msg.nClasses]).get(action)];
    }

    // one step
    let observation: Observation = await env.step([action])
    let done = observation.terminated ? 1 : 0
    let reward = Object.values(observation.reward_dict).reduce((s, v) => s + v, 0)
    let next_state = observation.state

    if (step >= max_episode_len - 1) {
      done = 1;
    }

    // calc td loss
    let tdloss: number
    if (msg.random_sample === 0) {
      let next_state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return T.fromArray(next_state).get(d) }));
      // let next_state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return T.fromArray(next_state).get(d)})));
      let next_state_input = new K.nn.Variable(await next_state_norm.reshape([1, msg.inputShape]).to(backend));
      let target_output = await model.call(next_state_input); // TODO: ターゲットモデルを使う
      let target_Q = T.max(T.min(T.cat([await target_output[0].data.to('cpu'), await target_output[1].data.to('cpu')], 0), 0)[0], 0)[0].get(0);
      let label = reward + (1 - done) * discount * target_Q

      tdloss = (Math.abs(label - y[0]) + Math.abs(label - y[1])) / 2;
    } else {
      tdloss = 1 // 最初のランダム探索の際は優先度は全て1
    }

    // save to buffer
    buffer_state.sets(state_norm, step);
    buffer_action.set(action, step);
    buffer_reward.set(reward, step);
    buffer_done.set(done, step);
    buffer_td.set(tdloss, step);

    state = T.fromArray(next_state);
    state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
    // state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));

    // if terminated, upload buffer and send message to the server
    if (done) {
      const buffer = new Map<string, T>();
      buffer.set('state', buffer_state.gets(new K.Slice(0, step + 1, 1)));
      buffer.set('action', buffer_action.gets(new K.Slice(0, step + 1, 1)));
      buffer.set('reward', buffer_reward.gets(new K.Slice(0, step + 1, 1)));
      buffer.set('done', buffer_done.gets(new K.Slice(0, step + 1, 1)));
      buffer.set('td', buffer_td.gets(new K.Slice(0, step + 1, 1)));

      await sendBlob(msg.buffer_id, new TensorSerializer().serialize(buffer));
      break;
    }
  }

  return true
}

async function compute_tester(msg: {
  type: String;
  inputShape: number;
  nClasses: number;
  weight_item_id: string;
  weight_target_item_id: string;
  env_id: string;
}) {

  writeTypeInfo(msg.type);
  writeBackendInfo(backend);

  // Load model weights
  let weight: Uint8Array
  try {
    weight = await recvBlob(msg.weight_item_id);
  } catch (e) {
    return [false, undefined]
  }
  let weights_load = new TensorDeserializer().deserialize(
    weight
  );
  for (const { name, parameter } of model.parametersWithName(true, false)) {
    parameter.data = await nonNull(weights_load.get(name)).to(backend);
    parameter.cleargrad();
  }

  // Initialize environment
  let env = await getEnv(msg.env_id);
  let state = T.fromArray((await env.reset()).state);
  let state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
  // let state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));
  const max_episode_len = 200; //todo

  // Initialize buffer
  let buffer_reward = T.zeros([max_episode_len, 1]);
  let sum_reward;

  // Start episode
  for (let step = 0; step < max_episode_len; step++) {

    // Action
    let state_input = new K.nn.Variable(await state_norm.reshape([1, msg.inputShape]).to(backend));
    let action = T.argmax((await (await model.call(state_input))[0].data.to('cpu')).reshape([msg.nClasses])).get(0);

    // one step
    let observation: Observation = await env.step([action]);
    console.log(observation)

    state = T.fromArray(observation.state);
    state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
    // state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));

    // save to buffer
    buffer_reward.set(Object.values(observation.reward_dict).reduce((s, v) => s + v, 0), step);

    // if terminated, upload buffer and send message to the server
    if (step >= max_episode_len - 1 || observation.terminated) {
      sum_reward = T.sum(buffer_reward.gets(new K.Slice(0, step + 1, 1))).get(0) // / (step+1); // ここをコメントアウトすると平均になる
      break;
    }
  }

  return [true, sum_reward]
}

let timer_start = Date.now(); // ミリ秒
let client_type: String;
async function run() {
  writeState('Connecting to distributed training server...');
  ws = new WebSocket(
    (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
    window.location.host +
    '/kakiage/ws'
  );
  writeState('Connecting to distributed training server... 3');
  ws.onopen = () => {
    writeState('Connected to server');
    ws.send(JSON.stringify({ "type": "worker" }));
  };
  ws.onclose = () => {
    writeState('Disconnected from server');
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    timer_start = Date.now();
    client_type = msg.type;

    if (msg.type === "reload") {
      window.location.reload();
    } else {
      writeIdInfo(msg.client_id)
      if (msg.type === "learner") {
        if (!model) {
          model = makeModel(msg.model, msg.inputShape, msg.nClasses);
          await model.to(backend);
        }
        if (!model_target) {
          model_target = makeModel(msg.model, msg.inputShape, msg.nClasses);
          await model_target.to(backend);
        }
        await K.tidy(async () => {
          await compute_learner(msg);
          return [];
        });
        ws.send(JSON.stringify({ "type": msg.type, "id": msg.grad }));
      } else if (msg.type === "actor") {
        if (!model && msg.random_sample === 0) {
          model = makeModel(msg.model, msg.inputShape, msg.nClasses);
          await model.to(backend);
        }
        await K.tidy(async () => {
          let success = await compute_actor(msg);
          ws.send(JSON.stringify({ "type": msg.type, "id": msg.buffer_id, "success": success }));
          return [];
        });
      } else if (msg.type === "tester") {
        if (!model) {
          model = makeModel(msg.model, msg.inputShape, msg.nClasses);
          await model.to(backend);
        }
        await K.tidy(async () => {
          let [success, sum_reward] = await compute_tester(msg);
          ws.send(JSON.stringify({ "type": msg.type, "id": msg.buffer_id, "success": success, "reward": sum_reward })); // TODO
          return [];
        });
      }
    }
  };
}

async function getBackend() {
  backend =
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    (
      document.querySelector(
        'input[name="backend"]:checked'
      )! as HTMLInputElement
    ).value as K.Backend;
  try {
    if (backend === 'webgl') {
      await K.tensor.initializeNNWebGLContext();
    }
    if (backend === 'webgpu') {
      await K.tensor.initializeNNWebGPUContext();
    }
  } catch (error) {
    alert(`Failed to initialize backend ${backend}. ${error}`);
    return;
  }
}

setInterval(() => {
  const millis = Date.now() - timer_start;
  console.log(`${Math.floor(millis / 1000)} since last onmessage`);
  if (millis > 30000) {
    console.log(`seconds elapsed = ${Math.floor(millis / 1000)}, msg.type = ${client_type}`);
    window.location.reload();
  }
}, 30000);

window.addEventListener('load', async () => {
  await getBackend();
  console.log(backend);
  await run();
});