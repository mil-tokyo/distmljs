import { KikyoConfig, Observation } from "../source/Kikyo_interface"
import { MujocoEnv } from "../source/Kikyo_Env"
import { InvertedDoublePendulum } from "./mujoco_env/invertedDoublePendulum"

function getOrCreateMujocoEnv(name: string, index?: number, config?: KikyoConfig): MujocoEnv {
  if (index == undefined) {
    index = 0
  }
  const key = name + "_" + index.toString()

  if (Kikyo.activeEnv[key] instanceof MujocoEnv) {
    // return cached env
    return Kikyo.activeEnv[key] as MujocoEnv
  } else {
    switch (name) {
      case "inverted_double_pendulum":
        return new InvertedDoublePendulum(index, config)

      default:
        return new MujocoEnv(name, index, [0], 1, 5, config)
    }
  }
}

const mujocoEnvsGetter: { [key: string]: (config?: KikyoConfig) => MujocoEnv } = {
  //   "singlePendulum": async () => {
  //     const env = getOrCreateMujocoEnv("MultiplePendulum", 0, 8, 1, { "Pendulums": 1 });
  //     return env
  //   },
  "Mujoco_InvertedDoublePendulum": (config?: KikyoConfig) => {
    return getOrCreateMujocoEnv("inverted_double_pendulum", 0, config = config);
  },
}

export { mujocoEnvsGetter }