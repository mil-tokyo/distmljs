import { waitUntil } from "./Kikyo/source/Kikyo_utility";

const testfunc = async ()=>{
  let ready:boolean = false;

  const buildUrl = "sources/mujoco";

  const bufferscript = document.createElement("script");
  bufferscript.src = buildUrl + "/buffer.js";
  bufferscript.onload = ()=>{console.log('bufferscript onload');ready=true;}
  document.body.appendChild(bufferscript);

  await waitUntil(() => { return ready })
  ready = false

  const preparescript = document.createElement("script");
  preparescript.src = buildUrl + "/mujoco_prepare.js";
  preparescript.onload = ()=>{console.log('preparescript onload');ready=true;}
  document.body.appendChild(preparescript);

  await waitUntil(() => { return ready })
  ready = false

  const loaderScript = document.createElement("script");
  loaderScript.src = buildUrl + "/mujoco_wasm.js";
  loaderScript.onload = ()=>{console.log('loaderScript onload');ready=true;}
  document.body.appendChild(loaderScript);

  await waitUntil(() => { return ready })
  ready = false

}

const testfunc2 = async ()=>{
  let ready:boolean = false;
  let loadfunc: any;
  // const buildUrl = "sources/mujoco2";
  const buildUrl = "http://localhost:8083/static/sources/mujoco2";
  // http://localhost:8083/static

  const window_props = Object.getOwnPropertyNames(window);

  const loaderScript = document.createElement("script");
  loaderScript.src = buildUrl + "/mujoco_wasm.js";
  loaderScript.type = 'module';
  loaderScript.onload = async ()=>{
    console.log('loaderScript onload');
    console.log((window as any).load_mujoco);
    console.log((window as any).mujoco_wasm);
    console.log((window as any).mujocoWasm);
    console.log(Object.getOwnPropertyNames(window).filter(v=>v.includes('joco')));
    const window_props2 = Object.getOwnPropertyNames(window);
    console.log(window_props.filter(v=>window_props2.includes(v)==false))
    console.log(window_props2.filter(v=>window_props.includes(v)==false))
    loadfunc=(window as any).load_mujoco;
    console.log(loadfunc)
    ready=true;
  }
  document.body.appendChild(loaderScript);

  await waitUntil(() => { return ready })
  ready = false;
  console.log('wait finish')
 
  const loaderScript2 = document.createElement("script");
  loaderScript2.src = buildUrl + "/mujoco_loader.js";
  loaderScript2.type = 'module';
  loaderScript2.onload = async ()=>{
    console.log('loaderScript2 onload');
    
    await waitUntil(() => { return (window as any).mujoco_ready })
    
    const mujoco = (window as any).mujoco
    var initialScene = "hammock.xml";
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
    mujoco.FS.writeFile("/working/" + initialScene, await(await fetch("./sources/mujoco2/" + initialScene)).text());
    

    ready=true;
  }
  document.body.appendChild(loaderScript2);


  // const module = await import(buildUrl + "/mujoco_wasm.js")
  // const {load_mujoco} = module;

  // const mujoco = await load_mujoco();


  // const mujoco = await loadfunc();
  // const mujoco_instance = await mujoco();


  // console.log(mujoco)
}

window.addEventListener('load', async () => {
  await testfunc2();
});