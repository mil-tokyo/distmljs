import { MujocoEnv } from "../../source/Kikyo_Env";
import { Observation } from "../../source/Kikyo_interface";

export class InvertedDoublePendulum extends MujocoEnv {
  constructor(index: number, config?: object) {
    super("inverted_double_pendulum", index, [0], 11, 5, config)
  }
  getObservation = async () => {
    if (this.simulation == undefined) {
      console.error("simulation is null")
      return {} as Observation
    }

    const observation: Observation = {
      terminated: false, state: [], reward_dict: {}
    }

    if (!observation.terminated && this.steps >= this.max_steps) {
      observation.terminated = true
    }

    const qpos = this.simulation.qpos
    const qvel = this.simulation.qvel
    const site_pos = this.simulation.site_xpos
    const constraint = this.simulation.qfrc_constraint

    console.log('qpos', qpos)
    console.log('tip', site_pos)
    console.log('constraint', constraint)

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

    if (!observation.terminated && site_pos[2] <= 1.0) {
      observation.terminated = true
    }

    observation.reward_dict['alive_bonus'] = 10;
    observation.reward_dict['distance_penalty'] = - (0.01 * site_pos[0] ** 2 + (site_pos[2] - 2) ** 2) // これあってるか？
    observation.reward_dict['velocity_penalty'] = - (1e-3 * observation.state[6] ** 2 + 5e-3 * observation.state[7] ** 2) //v2ってなんだ

    return observation
  }
  reset_camera(): void {
    this.renderer?.camera.position.set(0, 1, -5)
    this.renderer?.camera.lookAt(0, 1, 0)
  }
}