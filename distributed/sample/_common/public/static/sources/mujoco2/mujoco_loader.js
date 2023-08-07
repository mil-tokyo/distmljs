import { load_mujoco } from "./mujoco_wasm.js";
console.log("import load mujoco")
var mujoco_instance = await load_mujoco();
window.mujoco = mujoco_instance;
window.mujoco_ready = true;

export {mujoco_instance}