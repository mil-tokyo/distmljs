import load_mujoco from "./mujoco_wasm.js";
console.log("import load mujoco")
window.mujoco = await load_mujoco();
console.log("load_mujoco()")
window.mujoco_ready = true;