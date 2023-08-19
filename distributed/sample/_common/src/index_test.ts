import { waitForClick, waitForSecond, waitUntil } from "./Kikyo/source/Kikyo_utility";
import { MujocoEnv } from "./Kikyo/source/Kikyo_Env";
import { MujocoRenderer } from "./Kikyo/mujoco/mujocoRenderer";

const testfunc = async () => {
  let ready: boolean = false;

  const buildUrl = "sources/mujoco";

  const bufferscript = document.createElement("script");
  bufferscript.src = buildUrl + "/buffer.js";
  bufferscript.onload = () => { console.log('bufferscript onload'); ready = true; }
  document.body.appendChild(bufferscript);

  await waitUntil(() => { return ready })
  ready = false

  const preparescript = document.createElement("script");
  preparescript.src = buildUrl + "/mujoco_prepare.js";
  preparescript.onload = () => { console.log('preparescript onload'); ready = true; }
  document.body.appendChild(preparescript);

  await waitUntil(() => { return ready })
  ready = false

  const loaderScript = document.createElement("script");
  loaderScript.src = buildUrl + "/mujoco_wasm.js";
  loaderScript.onload = () => { console.log('loaderScript onload'); ready = true; }
  document.body.appendChild(loaderScript);

  await waitUntil(() => { return ready })
  ready = false

}

const testfunc2 = async () => {
  let ready: boolean = false;
  let loadfunc: any;
  const buildUrl = "sources/mujoco2";

  console.log('testfunc2');

  const loaderScript2 = document.createElement("script");
  loaderScript2.src = buildUrl + "/mujoco_loader.js";
  loaderScript2.type = 'module';
  loaderScript2.onload = async () => {
    console.log('loaderScript2 onload');

    await waitUntil(() => { return (window as any).mujoco_ready })

    const mujoco = (window as any).mujoco
    console.log('mujoco_ready');

    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

    console.log('mount');

    ready = true;
  }
  document.body.appendChild(loaderScript2);


  await waitUntil(() => { return ready })
  ready = false;
  console.log('wait finish');

  var initialScene = "humanoid.xml";
  const mujoco = (window as any).mujoco
  mujoco.FS.writeFile("/working/" + initialScene, await (await fetch("./sources/mujoco2/" + initialScene)).text());

  const model = new mujoco.Model("/working/" + initialScene);
  console.log(model)
  const state = new mujoco.State(model);
  console.log(state)
  const simulation = new mujoco.Simulation(model, state);
  console.log(simulation)

  console.log((window as any).mujoco.Simulation)
  await waitForClick('load done. click to start rendering')

  const ren = new MujocoRenderer()

  await ren.init(model, state, simulation, mujoco)
}

const testfunc3 = async () => {
  //test with mujoco env
  console.log("test 3")

  const env = new MujocoEnv('humanoid', 0, 0, 10, { 'visualize': true })

  await waitForClick("wait for clk 1");

  var o = await env.reset();
  console.log(o)

  await waitForClick("wait for clk 2");

  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.033);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");

  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");

  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");
  
  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");
  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.033);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");

  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");

  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");
  
  for (let i = 0; i < 100; i++) {
    await waitForSecond(0.01);
    o = await env.step([]);
    console.log(o)
    // await waitForClick("wait for clk ...");
  }
  await waitForClick("wait for clk 3");

  o = await env.reset();
  console.log(o)

  // await waitForClick("wait for clk 4");

  // for (let i = 0; i < 20; i++) {
  //   await env.step([]);
  //   sleep(100);
  // }
}
const testfunc4 = async () => {
  let ready: boolean = false;
  const buildUrl = "sources/mujoco2";
  const loaderScript2 = document.createElement("script");
  loaderScript2.src = buildUrl + "/mujoco_loader.js";
  loaderScript2.type = 'module';
  loaderScript2.onload = async () => {
    console.log('loaderScript2 onload');

    await waitUntil(() => { return (window as any).mujoco_ready })

    const mujoco = (window as any).mujoco
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

    ready = true;
  }
  document.body.appendChild(loaderScript2);

  await waitUntil(() => { return ready })
  ready = false;
  var initialScene = "humanoid.xml";
  const mujoco = (window as any).mujoco
  mujoco.FS.writeFile("/working/" + initialScene, await (await fetch("./sources/mujoco2/" + initialScene)).text());
  const model = new mujoco.Model("/working/" + initialScene);
  console.log(model)
  const state = new mujoco.State(model);
  console.log(state)
  const simulation = new mujoco.Simulation(model, state);
  console.log(simulation)

  console.log((window as any).mujoco.Simulation)


  await waitForClick('load done. click to start rendering')

  const ren = new MujocoRenderer()

  await ren.init(model, state, simulation, mujoco)

  // await waitForClick('renderer init done. click to step')
  let t = 0
  for (let i = 0; i < 90; i++) {
    t += 33
    ren.step(t)
    await waitForClick('click to simulate 1/30 sec')
  }

}


window.addEventListener('load', async () => {
  await testfunc3();
});