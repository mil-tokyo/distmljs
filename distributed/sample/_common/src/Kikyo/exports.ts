import { Env } from "./source/Kikyo_Env"
import { KikyoGlobal } from "./source/Kikyo"
import { Observation } from "./source/Kikyo_interface"
import { getEnv } from "./environments/env_loader"

window.Kikyo = new KikyoGlobal()

export { Env, Observation, getEnv }
export type { }