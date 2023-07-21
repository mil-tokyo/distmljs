import { Env, UnityEnv } from "../source/Kikyo_Env"

function getOrCreateUnityEnv(name: string, index: number, action_size: number, state_size: number): UnityEnv {
  const key = name + "_" + index.toString()
  if (Kikyo.activeEnv[key] instanceof UnityEnv) {
    // return cached env
    return Kikyo.activeEnv[key] as UnityEnv
  } else {
    return new UnityEnv(name, index, action_size, state_size)
  }
}

const unityEnvsGetter: { [key: string]: () => Promise<UnityEnv> } = {
  "SinglePendulum": async () => {
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 8, 1);
    await env.init({ "Pendulums": 1 })
    return env
  },
  "DoublePendulum": async () => {
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 14, 1);
    await env.init({ "Pendulums": 2 })
    return env
  },
}

export { unityEnvsGetter }