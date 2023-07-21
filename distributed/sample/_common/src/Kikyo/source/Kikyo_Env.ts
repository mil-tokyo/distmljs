import { KikyoUnityMethod, Observation, SendValue } from "./Kikyo_interface";

abstract class Env {
    name: string;
    initialized: boolean;
    state_size: number;
    action_size: number;

    constructor(name: string, action_size: number, state_size: number) {
        this.name = name
        this.action_size = action_size;
        this.state_size = state_size;
        this.initialized = false
        Kikyo.activeEnv[name] = this
    }

    abstract _init(config?: object): Promise<Observation>;
    abstract _step(action: number[]): Promise<Observation>;
    abstract _reset(): Promise<Observation>;

    async init(config?: object): Promise<Observation> {
        var ret = await this._init();
        this.initialized = true;
        return ret;
    }
    async step(action: number[]): Promise<Observation> {
        if (this.initialized == false) {
            console.error("env not initialised. please call env.init() before calling env.step()")
        }
        if (action.length != this.action_size) {
            console.log("!! action length is wrong !!")
        }
        return await this._step(action);
    }
    async reset(): Promise<Observation> {
        if (this.initialized == false) {
            console.error("env not initialised. please call env.init() before calling env.reset()")
        }
        return await this.reset();
    }
}

class UnityEnv extends Env {
    index: number;

    constructor(name: string, index: number, action_size: number, state_size: number) {
        super(name + "_" + index.toString(), action_size, state_size);
        this.index = index
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
            const sendValue: SendValue = { EnvName: this.name, Index: this.index, Token: token, Action: action, Config: config }
            Kikyo.getOrCreateUnityInstance().then(u => u.SendMessage('KikyoManager', method, JSON.stringify(sendValue)));
            console.log("SendValue:" + JSON.stringify(sendValue))
        });
    }

    // todo: handle multiple call -> resetとConstructerだけでいいのでは？
    async _init(config?: object): Promise<Observation> {
        return this.UnityPromise('CreateEnvironment', undefined, config)
    }

    async _step(action: number[]): Promise<Observation> {
        return this.UnityPromise('StepEnvironment', action)
    }

    async _reset(): Promise<Observation> {
        return this.UnityPromise('ResetEnvironment')
    }
}

export { Env, UnityEnv }