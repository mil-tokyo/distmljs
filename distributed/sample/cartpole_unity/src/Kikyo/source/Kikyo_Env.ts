import { KikyoUnityMethod, Observation, SendValue } from "./Kikyo_interface";

class Env {
    name: string;
    index: number;
    initialized: boolean;

    constructor(name: string, index: number) {
        this.name = name
        this.index = index
        this.initialized = false
        Kikyo.activeEnv[name + "_" + index.toString()] = this
    }

    UnityPromise(method: KikyoUnityMethod, action?: number[]): Promise<Observation> {
        return new Promise<Observation>((resolve) => {
            const token = Math.random().toString(32).substring(2)
            console.log('send CreateEnvironment with token:' + token)
            Kikyo.callback[token] = (observation: Observation) => {
                console.log(observation);
                resolve(observation);
                delete Kikyo.callback[token]
                console.log('finish promise of createAsync with token:' + token)
            }
            const sendValue: SendValue = { EnvName: this.name, Index: this.index, Token: token, Action: action }
            Kikyo.getOrCreateUnityInstance().then(u => u.SendMessage('KikyoManager', method, JSON.stringify(sendValue)));
            console.log("SendValue:" + JSON.stringify(sendValue))
        });
    }

    async init(): Promise<Observation> {
        return this.UnityPromise('CreateEnvironment')
            .then(r => {
                this.initialized = true;
                return r
            })
    }

    async step(action: number[]): Promise<Observation> {
        if (this.initialized == false) {
            console.error("env not initialised. please call env.init() before calling env.step()")
        }
        return this.UnityPromise('StepEnvironment', action = action)
    }

    async reset(): Promise<Observation> {
        if (this.initialized == false) {
            return this.init()
        }
        return this.UnityPromise('ResetEnvironment')
    }

}
export { Env }