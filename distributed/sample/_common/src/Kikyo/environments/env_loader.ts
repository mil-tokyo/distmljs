import { unityEnvsGetter } from "../environments/unityEnv";
import { Env } from "../source/Kikyo_Env";

async function getEnv(id: string): Promise<Env> {
  if (id in Object.keys(unityEnvsGetter)) {
    return await unityEnvsGetter[id]() as Env
  }

  return null as unknown as Env;
}

export { getEnv }