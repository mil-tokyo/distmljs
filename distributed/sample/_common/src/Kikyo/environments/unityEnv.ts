import { Env, UnityEnv } from "../source/Kikyo_Env"

function getOrCreateUnityEnv(name: string, index: number, action_size: number, state_size: number, config?: object): UnityEnv {
  const key = name + "_" + index.toString()
  if (Kikyo.activeEnv[key] instanceof UnityEnv) {
    // return cached env
    return Kikyo.activeEnv[key] as UnityEnv
  } else {
    return new UnityEnv(name, index, action_size, state_size, config)
  }
}

const unityEnvsGetter: { [key: string]: () => Promise<UnityEnv> } = {
  "singlePendulum": async () => {
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 8, 1, { "Pendulums": 1 });
    return env
  },
  "doublePendulum": async () => {
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 14, 1, { "Pendulums": 2 });
    return env
  },
}

export { unityEnvsGetter }