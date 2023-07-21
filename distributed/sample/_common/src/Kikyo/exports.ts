import { Env } from "./source/Kikyo_Env"
import { Kikyo } from "./source/Kikyo"
import { Observation } from "./source/Kikyo_interface"
import { getEnv } from "./environments/env_loader"

window.Kikyo = Kikyo

export { Kikyo, Env, Observation, getEnv }
export type { }