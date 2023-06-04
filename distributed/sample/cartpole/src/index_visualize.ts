import { min, throttle } from 'lodash';
import * as K from 'kakiage';
import T = K.tensor.CPUTensor;
import TensorDeserializer = K.tensor.TensorDeserializer;
import TensorSerializer = K.tensor.TensorSerializer;
import F = K.nn.functions;
import L = K.nn.layers;
import { Env } from './cartpole';
import { makeModel } from './models';
import { tensorTextureShapeFormatDefault } from '../../../../dist/tensor/webgl/webglTensor';

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
  const force_value = await fetch(`/kakiage/blob/${itemId}`, { method: 'GET' });
  if (!force_value.ok) {
    throw new Error('Server response to save is not ok');
  }
  const resp = await force_value.arrayBuffer();
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
  let env = new Env();
  let state = T.fromArray(env.reset(true, false));
  let ep_reward: number;
  const max_episode_len = env.max_episode_len;

  // for visualization
  const canvas = document.getElementById("canvas") as HTMLCanvasElement | null;
  const field_w: number = 160; // half size
  const field_h: number = 160; // half size
  const cart_w = 20; // half size
  const cart_h = 10; // half size
  const scale = 100;
  const pole_len = scale * env.pole_length;
  let force_value = 20; // TODO: env.forceを使う

  btn_flag = true;
  while (btn_flag) {
    
    console.log('start new episode.')
    state = T.fromArray(env.reset(true, false));
    ep_reward = 0;

    // Start episode
    for (let step=0; step<max_episode_len; step++) {
      
      // Action
      let rnd_sample = Math.random();
      let rnd_move = Number(rnd_sample < epsilon)
      let action: number = 0;
      if (rnd_move) {
        let rand_tmp = Math.random()
        action = Math.floor(rand_tmp * msg.nClasses);
      } else {
        let state_input = new K.nn.Variable(await state.reshape([1, msg.inputShape]).to(backend));
        action = T.argmax((await (await model.call(state_input))[0].data.to('cpu')).reshape([msg.nClasses])).get(0);
      }

      // one step
      let [[x, x_dot, theta, theta_dot], reward, done]: 
        [number[], number, number] = env.step(
          [state.get(0), state.get(1), state.get(2), state.get(3)] // 同期用のコードのため、状態を入力するようになっている
          , action
        );
      state = T.fromArray([x, x_dot, theta, theta_dot]);
      ep_reward += reward;
      
      // 正負で桁がずれること嫌ったが、もっとましな実装にする
      if (reward < 0) {
        writeRewardInfo("- "+String(-reward.toFixed(2)));
      } else {
        writeRewardInfo("+"+String(reward.toFixed(2)));
      }

      if (ep_reward < 0) {
        writeEpRewardInfo("- "+String(-ep_reward.toFixed(2)));
      } else {
        writeEpRewardInfo("+"+String(ep_reward.toFixed(2)));
      }
      



      if (canvas !== null && canvas.getContext) {
        const ctx = canvas.getContext("2d"); //2次元描画
        if (ctx !== null) {

          // initialize field
          ctx.clearRect(0, 0, field_w*2, field_h*2);

          // draw horizontal rail
          ctx.beginPath();
          ctx.moveTo(0, field_h);
          ctx.lineTo(field_w*2, field_h);
          ctx.strokeStyle = "#d3d3d3";
          ctx.lineWidth = 3;
          ctx.stroke();

          // draw cart
          ctx.fillStyle = '#000000';
          ctx.fillRect(field_w+x*scale-cart_w, field_h-cart_h, cart_w*2, cart_h*2);

          // draw pole
          ctx.beginPath();
          ctx.moveTo(field_w+x*scale, field_h);
          ctx.lineTo(field_w+x*scale+pole_len*Math.sin(theta), field_h-pole_len*Math.cos(theta));
          ctx.strokeStyle = "#cd853f";
          ctx.lineWidth = 10;
          ctx.stroke();


          // draw joint 
          ctx.fillStyle = '#c0c0c0';
          ctx.beginPath();
          ctx.arc(field_w+x*scale, field_h, 4, 0, Math.PI*2, false);
          ctx.fill();

          // draw force
          let direction;
          ctx.beginPath();
          if (action === 0) {
            direction = -1;
          } else {
            direction = +1;
          }
          force_value = 10;
          ctx.moveTo(field_w+x*scale-force_value, field_h+cart_h*2);
          ctx.lineTo(field_w+x*scale+force_value, field_h+cart_h*2);
          ctx.moveTo(field_w+x*scale+direction*(force_value-5), field_h+cart_h*2-5);
          ctx.lineTo(field_w+x*scale+direction*(force_value), field_h+cart_h*2);
          ctx.moveTo(field_w+x*scale+direction*(force_value-5), field_h+cart_h*2+5);
          ctx.lineTo(field_w+x*scale+direction*(force_value), field_h+cart_h*2);
          ctx.strokeStyle = "#000000";
          ctx.lineWidth = 2;
          ctx.stroke();

        }
      }

      await sleep(20);

      if (!btn_flag) {
        break
      }

      if (done || step >= max_episode_len-1) {
        break
      }
    }
  }
  ws.send(JSON.stringify({"type":"visualizer"}));
  return []
}

let btn_flag = false
function set_button() {
  let btn = document.getElementById('btn');
  if (btn !== null) {
    btn.addEventListener('click', function() {
      if (!btn_flag) {
        ws.send(JSON.stringify({"type":"visualizer"}));
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
    const rangeOnChange = (e: Event) =>{
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
      randbtn.addEventListener('click', function() {
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