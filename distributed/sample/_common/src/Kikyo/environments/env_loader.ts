import { unityEnvsGetter } from "../environments/unityEnv";
import { Env } from "../source/Kikyo_Env";

async function getEnv(id: string): Promise<Env> {
  console.log("loading env:" + id)
  if (id in unityEnvsGetter) {
    return await unityEnvsGetter[id]() as Env
  }

  alert(`env missing. please ensure env_id is correct. id=${id}, possible_ids=${Object.keys(unityEnvsGetter)}`);
  return null as unknown as Env;
}

export { getEnv }