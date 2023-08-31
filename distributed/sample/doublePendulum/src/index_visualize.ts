import { throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;
import F = K.nn.functions;
import { makeModel } from './models';
import { getEnv } from "./Kikyo/exports";

let ws: WebSocket;
let backend: K.Backend = 'cpu';

const MAX_EPISODE_LEN = 200;

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const writeState = throttle((message: string) => {
  document.getElementById('state')!.innerText = message;
}, 1000);

const writeLoadStatusInfo = throttle((message: string) => {
  document.getElementById('table-loadstatus')!.innerText = message;
}, 1000);

const writeTypeInfo = throttle((type: String) => {
  document.getElementById('table-type')!.innerText = type.toString();
}, 1000);

const writeRewardInfo = throttle((type: String) => {
  document.getElementById('table-reward')!.innerText = type.toString();
}, 10);

const writeEpRewardInfo = throttle((type: String) => {
  document.getElementById('table-ep-reward')!.innerText = type.toString();
}, 10);

const writeIdInfo = throttle((type: String) => {
  document.getElementById('table-id')!.innerText = type.toString();
}, 1000);

async function recvBlob(itemId: string): Promise<Uint8Array> {
  const f = await fetch(`/kakiage/blob/${itemId}`, { method: 'GET' });
  if (!f.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await f.arrayBuffer();
  return new Uint8Array(resp);
}

let model: K.nn.core.Layer;

async function compute_visualizer(msg: {
  type: String;
  env_key: string;
  inputShape: number;
  nClasses: number;
  weight_actor_item_id: string;
}) {
  writeTypeInfo('visualizer');
  console.log('visualizer episode start')

  // Load model weights
  let weight: Uint8Array
  try {
    weight = await recvBlob(msg.weight_actor_item_id);
  } catch (e) {
    writeLoadStatusInfo('Failed. Try again.');
    return false
  }
  writeLoadStatusInfo('Successfully loaded.');
  const weights_actor = new TensorDeserializer().deserialize(
    weight
  );
  for (const { name, parameter } of model.parametersWithName(true, false)) {
    parameter.data = await nonNull(weights_actor.get(name)).to(backend);
    parameter.cleargrad();
  }

  // Initialize environment
  const env = await getEnv('Mujoco_InvertedDoublePendulum');
  let { state, terminated, reward_dict } = await env.reset();
  let stateTensor = T.fromArray(state);
  let reward = Object.values(reward_dict).reduce((partialSum, a) => partialSum + a, 0);
  let ep_reward: number;

  btn_flag = true;
  while (btn_flag) {

    console.log('start new episode.');
    ({ state, terminated, reward_dict } = await env.reset());
    stateTensor = T.fromArray(state);
    reward = Object.values(reward_dict).reduce((partialSum, a) => partialSum + a, 0);

    ep_reward = 0;

    // Start episode
    for (let step = 0; step < MAX_EPISODE_LEN; step++) {

      // Action
      let rnd_sample = Math.random();
      let rnd_move = Number(rnd_sample < epsilon);
      let action: number;
      if (rnd_move) {
        let rand_tmp = Math.random();
        action = Math.floor(rand_tmp * msg.nClasses);
      } else {
        let state_input = new K.nn.Variable(await stateTensor.reshape([1, msg.inputShape]).to(backend));
        action = T.argmax((await (await model.call(state_input))[0].data.to('cpu')).reshape([msg.nClasses])).get(0);
      }

      // one step
      let env_action: number = 0;
      switch (action) {
        case 0:
          env_action = - 0.5;
        case 1:
          env_action = + 0.5;
      };

      ({ state, terminated, reward_dict } = await env.step([env_action]));
      stateTensor = T.fromArray(state);
      reward = Object.values(reward_dict).reduce((partialSum, a) => partialSum + a, 0);
      ep_reward += reward;

      if (reward < 0) {
        writeRewardInfo("- " + String(-reward.toFixed(2)));
      } else {
        writeRewardInfo("+" + String(reward.toFixed(2)));
      }

      if (ep_reward < 0) {
        writeEpRewardInfo("- " + String(-ep_reward.toFixed(2)));
      } else {
        writeEpRewardInfo("+" + String(ep_reward.toFixed(2)));
      }

      await sleep(30);

      if (!btn_flag) {
        break
      }

      if (terminated || step >= MAX_EPISODE_LEN - 1) {
        break
      }
    }
  }
  ws.send(JSON.stringify({ "type": "visualizer" }));
  return []
}

let btn_flag = false
function set_button() {
  let btn = document.getElementById('btn');
  if (btn !== null) {
    btn.addEventListener('click', function () {
      if (!btn_flag) {
        ws.send(JSON.stringify({ "type": "visualizer" }));
      }
      btn_flag = false
    })
  }
  return btn
}


let epsilon = 0
function set_random_slider() {
  const slider = <HTMLInputElement>document.getElementById('random');
  const txt = <HTMLInputElement>document.getElementById('current-random-value');
  if (slider !== null && txt !== null) {
    const setCurrentValue = (val: string) => {
      let toprint = ''
      if (val.length === 1) {
        toprint = `${val}.00`;
      } else if (val.length === 3) {
        toprint = `${val}0`;
      } else {
        toprint = val;
      }
      txt.innerText = `ランダム行動選択の割合： ${toprint}`;
    }
    const rangeOnChange = (e: Event) => {
      const { target } = e;
      if (!(target instanceof HTMLInputElement)) {
        return
      }
      epsilon = Number(slider.value);
      setCurrentValue(target.value);
    }
    slider.addEventListener('input', rangeOnChange);
    setCurrentValue(slider.value);

    // init_random button
    let randbtn = document.getElementById('init-random');
    if (randbtn !== null) {
      randbtn.addEventListener('click', function () {
        slider.value = '0';
        setCurrentValue(slider.value);
      })
    }
  }
}


async function update_stats(msg: {
  type: String;
  iters: number;
  bfsize: number;
  reward_mean: number;
  reward_std: number;
  time: number;
}) {
  console.log(msg);
}

async function run() {
  writeState('Connecting to distributed training server...');
  ws = new WebSocket(
    (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
    window.location.host +
    '/kakiage/ws'
  );
  ws.onopen = async () => {
    writeState('Connected to server');
    console.log('send hello');
    let btn = set_button();
    set_random_slider();
    setTimeout(function () {
      let clickevent = new Event('click');
      if (btn !== null) {
        btn.dispatchEvent(clickevent)
      };
    }, 1000)
  };
  ws.onclose = () => {
    writeState('Disconnected from server');
    btn_flag = false
  };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    writeIdInfo(msg.client_id)
    if (msg.type === "visualizer.weight") {
      if (!model) {
        console.log('making model')
        model = makeModel(msg.model, msg.inputShape, msg.nClasses);
        console.log('finish making model')
        await model.to(backend);
      }
      await K.tidy(async () => {
        await compute_visualizer(msg);
        return [];
      });
    } else if (msg.type === "visualizer.stats") {
      update_stats(msg);
    }
  };
}

window.addEventListener('load', async () => {
  backend = (new URLSearchParams(window.location.search).get('backend') ||
    'webgl') as K.Backend;
  if (backend === 'webgl') {
    await K.tensor.initializeNNWebGLContext();
  }
  await run();
});