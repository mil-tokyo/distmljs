import { unityEnvsGetter } from "../environments/unityEnv";
import { Env } from "../source/Kikyo_Env";
import { KikyoConfig } from "../source/Kikyo_interface";
import { mujocoEnvsGetter } from "./mujocoEnv";

async function getEnv(id: string, config?: KikyoConfig): Promise<Env> {
  console.log("loading env:" + id)
  if (id in unityEnvsGetter) {
    return unityEnvsGetter[id](config) as Env
  }
  if (id in mujocoEnvsGetter) {
    return mujocoEnvsGetter[id](config) as Env
  }

  alert(`env missing. please ensure env_id is correct. id=${id}, possible_ids=${Object.keys(unityEnvsGetter)}`);
  return null as unknown as Env;
}

export { getEnv }