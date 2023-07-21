import { KikyoUnityMethod, Observation, SendValue } from "./Kikyo_interface";
import { waitForClick } from "./Kikyo_utility";

abstract class Env {
    name: string;
    state_size: number;
    action_size: number;
    config: object;

    constructor(name: string, action_size: number, state_size: number, config?: object) {
        this.name = name
        this.action_size = action_size;
        this.state_size = state_size;
        if (config == undefined) {
            config = {}
        }
        this.config = config;
        Kikyo.activeEnv[name] = this
    }

    set_config(config: object) {
        this.config = { ...this.config, ...config };
    }

    async step(action: number[]): Promise<Observation> {
        if (action.length != this.action_size) {
            console.log("!! action length is wrong !!")
        }
        console.log("please override")
        return { "state": Array<number>(this.state_size), "reward_dict": {}, "terminated": false };
    }
    async reset(): Promise<Observation> {
        console.log("please override")
        return { "state": Array<number>(this.state_size), "reward_dict": {}, "terminated": false };
    }
}

class UnityEnv extends Env {
    index: number;
    envName: string;

    constructor(envName: string, index: number, action_size: number, state_size: number, config?: object) {
        super(envName + "_" + index.toString(), action_size, state_size, config);
        this.index = index
        this.envName = envName
        // example... envName: cartpole, index: 1, name: cartpole_1
    }

    UnityPromise(method: KikyoUnityMethod, action?: number[], config?: object): Promise<Observation> {
        return new Promise<Observation>((resolve) => {
            const token = Math.random().toString(32).substring(2)
            console.log('send CreateEnvironment with token:' + token)
            Kikyo.callback[token] = (observation: Observation) => {
                console.log(observation);
                resolve(observation);
                delete Kikyo.callback[token]
                console.log('finish promise of createAsync with token:' + token)
            }
            const sendValue: SendValue = { EnvName: this.envName, Index: this.index, Token: token, Action: action, Config: config }
            console.log("before SendValue:" + JSON.stringify(sendValue))
            Kikyo.getOrCreateUnityInstance().then(u => {
                console.log(`Kikyo.getOrCreateUnityInstance done. sending message...${method}, ${sendValue}`)
                console.log(sendValue)
                u.SendMessage('KikyoManager', method, JSON.stringify(sendValue))
                console.log(`SendMessage done. sending message...${method}, ${sendValue}`)
            });
            console.log("SendValue:" + JSON.stringify(sendValue))
        });
    }

    // // todo: handle multiple call -> resetとConstructerだけでいいのでは？
    // async init(config?: object): Promise<Observation> {
    //     return this.UnityPromise('CreateEnvironment', undefined, config)
    // }

    async step(action: number[]): Promise<Observation> {
        await waitForClick(`waiting click... step:${action}`)
        return await this.UnityPromise('StepEnvironment', action)
    }

    async reset(): Promise<Observation> {
        console.log("aaaaaaaaaaaaa")
        await waitForClick(`waiting click... reset:${this.config}`)
        return await this.UnityPromise('ResetEnvironment', undefined, this.config)
    }
}

export { Env, UnityEnv }