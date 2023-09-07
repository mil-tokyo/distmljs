import { Env, UnityEnv } from "../source/Kikyo_Env"
import { KikyoConfig } from "../source/Kikyo_interface"

function getOrCreateUnityEnv(name: string, index: number, action_size: number, state_size: number, config?: KikyoConfig): UnityEnv {
  const key = name + "_" + index.toString()
  if (Kikyo.activeEnv[key] instanceof UnityEnv) {
    // return cached env
    return Kikyo.activeEnv[key] as UnityEnv
  } else {
    return new UnityEnv(name, index, action_size, state_size, config)
  }
}

const unityEnvsGetter: { [key: string]: (config?: KikyoConfig) => UnityEnv } = {
  "singlePendulum": (config?: KikyoConfig) => {
    if (config == undefined) { config = {} }
    config.Pendulums = 1
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 8, 1, config);
    return env
  },
  "doublePendulum": (config?: KikyoConfig) => {
    if (config == undefined) { config = {} }
    config.Pendulums = 2
    const env = getOrCreateUnityEnv("MultiplePendulum", 0, 14, 1, config);
    return env
  },
}

export { unityEnvsGetter }