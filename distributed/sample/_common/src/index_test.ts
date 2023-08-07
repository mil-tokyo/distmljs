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

window.addEventListener('load', async () => {
  await testfunc();
});