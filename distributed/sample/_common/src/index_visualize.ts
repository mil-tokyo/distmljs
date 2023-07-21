import { min, throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;
import F = K.nn.functions;
import L = K.nn.layers;
import { Kikyo, getEnv } from './Kikyo/exports';
import { Observation } from './Kikyo/source/Kikyo_interface';
import { makeModel } from './models';
import { config } from 'chai';

let ws: WebSocket;
let backend: K.Backend = 'cpu';

function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
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

async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let model: K.nn.core.Layer;

async function compute_visualizer(msg: {
  type: String;
  inputShape: number;
  nClasses: number;
  weight_actor_item_id: string;
  env_id: string;
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
  let env = await getEnv(msg.env_id);
  env.set_config({ EnableView: true })
  // let state = T.fromArray(env.reset(true, false));
  // let state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));
  let state: T, state_norm: T;
  let ep_reward: number;
  const max_episode_len = 200;

  // for visualization
  // const maze_canvas = document.getElementById("maze_canvas") as HTMLCanvasElement | null;
  // const maze_field_len: number = 320;
  // const h: number = Math.floor(maze_field_len / env.maze.length); // y
  // const w: number = Math.floor(maze_field_len / env.maze[0].length); // x
  // const b: number = Math.min(h,w)
  // const radius: number = env.radius * b

  btn_flag = true;
  while (btn_flag) {

    console.log('start new episode.')
    state = T.fromArray(((await env.reset()).state));
    state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
    // state_norm = T.fromArray(env.normalize([...Array(6).keys()].map((d) => {return state.get(d)})));
    ep_reward = 0;

    // Start episode
    for (let step = 0; step < max_episode_len; step++) {

      // Action
      let rnd_sample = Math.random();
      let rnd_move = Number(rnd_sample < epsilon)
      let action: number = 0;
      if (rnd_move) {
        let rand_tmp = Math.random()
        action = Math.floor(rand_tmp * msg.nClasses);
      } else {
        let state_input = new K.nn.Variable(await state_norm.reshape([1, msg.inputShape]).to(backend));
        action = T.argmax((await (await model.call(state_input))[0].data.to('cpu')).reshape([msg.nClasses])).get(0);
      }

      // one step
      let observation: Observation = await env.step([action]);
      let reward = Object.values(observation.reward_dict).reduce((s, v) => s + v, 0)
      let done = observation.terminated ? 1 : 0

      state = T.fromArray(observation.state);
      state_norm = T.fromArray([...Array(msg.inputShape).keys()].map((d) => { return state.get(d) }));
      // state_norm = T.fromArray(env.normalize([x, y, x_dot, y_dot, goal_x, goal_y]));
      ep_reward = ep_reward + reward;

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

      // if (maze_canvas !== null && maze_canvas.getContext) {
      //   const maze_context = maze_canvas.getContext("2d"); //2次元描画
      //   if (maze_context !== null) {

      //     // initialize field
      //     maze_context.clearRect(0,0,maze_field_len,maze_field_len);

      //     maze_context.fillStyle = "#a9a9a9";
      //     maze_context.beginPath();
      //     maze_context.moveTo(0, 0);
      //     maze_context.lineTo(200, 200);
      //     maze_context.strokeStyle = "red";
      //     maze_context.lineWidth = 10;
      //     maze_context.stroke();

      //     maze_context.fillStyle = "#a9a9a9";
      //     for (let i=0; i<env.maze.length; i++) { // y
      //       for (let j=0; j<env.maze[0].length; j++) { // x
      //         if (env.maze[i][j] == 'x') {
      //           maze_context.fillRect(j * b, i * b, b, b);
      //         }
      //       }
      //     }

      //     // draw goal
      //     maze_context.fillStyle = 'red';
      //     maze_context.beginPath();
      //     maze_context.arc(env.goal_x * b, env.goal_y * b, radius, 0, Math.PI*2, false);
      //     maze_context.fill();

      //     // draw player
      //     if (env.collide) {
      //       maze_context.fillStyle = '#daa520';
      //     } else {
      //       maze_context.fillStyle = 'blue';
      //     }
      //     maze_context.beginPath();
      //     maze_context.arc(x * b, y * b, radius, 0, Math.PI*2, false);
      //     maze_context.fill();

      //     // draw force
      //     maze_context.beginPath();
      //     maze_context.moveTo(x * b, y * b) ;
      //     maze_context.lineTo(x * b + env.x_acc * 20, y * b + env.y_acc * 20)
      //     if (rnd_move) {
      //       maze_context.strokeStyle = '#ff0000';
      //     } else {
      //       maze_context.strokeStyle = '#87ceeb';
      //     }
      //     maze_context.lineWidth = 5;
      //     maze_context.stroke();


      //   }
      // }
      await sleep(50);

      if (!btn_flag) {
        break
      }

      if (done || step >= max_episode_len - 1) {
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
  ws.onopen = () => {
    writeState('Connected to server');
    console.log('send hello');
    set_button();
    set_random_slider();
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