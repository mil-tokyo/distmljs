import { Observation } from "../source/Kikyo_interface"
import { MujocoEnv } from "../source/Kikyo_Env"

function getOrCreateMujocoEnv(name: string, index: number, action_map: [number], state_size: number, config?: object): MujocoEnv {
  const key = name + "_" + index.toString()
  if (Kikyo.activeEnv[key] instanceof MujocoEnv) {
    // return cached env
    return Kikyo.activeEnv[key] as MujocoEnv
  } else {
    return new MujocoEnv(name, index, action_map.length, state_size, action_map, config)
  }
}

const mujocoEnvsGetter: { [key: string]: () => Promise<MujocoEnv> } = {
//   "singlePendulum": async () => {
//     const env = getOrCreateMujocoEnv("MultiplePendulum", 0, 8, 1, { "Pendulums": 1 });
//     return env
//   },
  "Mujoco_InvertedDoublePendulum": async () => {
    const env = getOrCreateMujocoEnv("inverted_double_pendulum", 0, [0], 11, {'visualize':true});
    env.getObservation = async () => {
        if (env.simulation == undefined) {
            console.error("simulation is null")
            return {} as Observation
        }

        const observation: Observation = {
            terminated: false, state: [], reward_dict: {}
        }

        if(!observation.terminated && env.steps >= env.max_steps){
            observation.terminated = true
        }

        const qpos = env.simulation.qpos
        const qvel = env.simulation.qvel
        const site_pos = env.simulation.site_xpos
        const constraint = env.simulation.qfrc_constraint

        console.log('qpos',qpos)
        console.log('tip',site_pos)
        console.log('constraint',constraint)

        observation.state.push(qpos[0])
        observation.state.push(Math.sin(qpos[1]))
        observation.state.push(Math.sin(qpos[2]))
        observation.state.push(Math.cos(qpos[1]))
        observation.state.push(Math.cos(qpos[2]))
        observation.state.push(qvel[0])
        observation.state.push(qvel[1])
        observation.state.push(qvel[2])
        observation.state.push(constraint[0])
        observation.state.push(constraint[1])
        observation.state.push(constraint[2])

        if(!observation.terminated && site_pos[2]<=1.0){
            observation.terminated = true
        }

        observation.reward_dict['alive_bonus'] = 10;
        observation.reward_dict['distance_penalty'] = - (0.01 * site_pos[0]**2 + (site_pos[2]-2)**2) // これあってるか？
        observation.reward_dict['velocity_penalty'] = - (1e-3 * observation.state[6]**2 + 5e-3 * observation.state[7]**2) //v2ってなんだ

        return observation
    }
    return env
  },
}

export { mujocoEnvsGetter }