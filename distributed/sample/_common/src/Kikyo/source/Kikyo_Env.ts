import { Model, Simulation, State } from "../declaration/mujoco_wasm";
import { KikyoGlobal } from "./Kikyo";
import { KikyoUnityMethod, Observation, SendValue } from "./Kikyo_interface";
import { MujocoRenderer } from "../mujoco/mujocoRenderer";

window.Kikyo = window.Kikyo ?? new KikyoGlobal()

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
            Kikyo.callback[token] = (observation: Observation) => {
                console.log(observation);
                resolve(observation);
                delete Kikyo.callback[token]
            }
            const sendValue: SendValue = { EnvName: this.envName, Index: this.index, Token: token, Action: action, Config: config }
            Kikyo.unity.getOrCreateInstance().then(u => {
                u.SendMessage('KikyoManager', method, JSON.stringify(sendValue))
            });
        });
    }

    async step(action: number[]): Promise<Observation> {
        return await this.UnityPromise('StepEnvironment', action)
    }

    async reset(): Promise<Observation> {
        return await this.UnityPromise('ResetEnvironment', undefined, this.config)
    }
}

class MujocoEnv extends Env {
    index: number;
    envName: string;
    sceneFile: string;
    model: Model | undefined;
    state: State | undefined;
    simulation: Simulation | undefined;
    renderer: MujocoRenderer | undefined;
    time: number;

    constructor(envName: string, index: number, action_size: number, state_size: number, config?: {[key:string]:any}) {
        super(envName + "_" + index.toString(), action_size, state_size, config);
        this.index = index
        this.envName = envName
        this.sceneFile = envName + ".xml"
        if(config && config.visualize==true){
            this.renderer = new MujocoRenderer()
        }
        this.time = 0
        // example... envName: cartpole, index: 1, name: cartpole_1
    }

    async getObservation(): Promise<Observation> {
        if (this.simulation == undefined) {
            console.error("simulation is null")
            return {} as Observation
        }

        const observation: Observation = {
            terminated: false, state: [], reward_dict: {}
        }

        observation.state.push(...Array.from(this.simulation.xpos))

        return observation
    }

    async step(action: number[]): Promise<Observation> {
        if (this.simulation == undefined) {
            console.error("simulation is null")
            return {} as Observation
        }

        const f = 1000000;
        const force = { x: Math.random()*f, y: Math.random()*f, z: Math.random()*f }
        const point = { x: 0, y: 0, z: 0 }
        // const point = { x: Math.random(), y: Math.random(), z: Math.random() }
        const bodyID = 0;

        this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);
        // todo: forceのやり方を適切に
        
        const dt = 33
        const timeMS = this.time + dt
        let timestep = this.simulation.model().getOptions().timestep;
        while (this.time < timeMS) {
            for (let i = 0; i < this.simulation.qfrc_applied.length; i++) { this.simulation.qfrc_applied[i] = 0.0; }
            // this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);
            this.simulation.step();
            this.time += timestep * 1000.0;
        }

        return await this.getObservation()
    }

    async reset(): Promise<Observation> {
        //todo: vis用のMujocoRootのReset
        const mujoco = await Kikyo.mujoco.getOrCreateInstance();
        
        mujoco.FS.writeFile("/working/" + this.sceneFile, await (await fetch("./sources/mujoco2/" + this.sceneFile)).text());

        console.log("/working/" + this.sceneFile)
        
        this.model = new mujoco.Model("/working/" + this.sceneFile);
        console.log(this.model)
        this.state = new mujoco.State(this.model);
        console.log(this.state)
        this.simulation = new mujoco.Simulation(this.model, this.state);
        console.log(this.simulation)
        
        if(this.renderer){
            await this.renderer.init(this.model,this.state,this.simulation)
        }
        this.time = 0

        return await this.getObservation()
    }
}
export { Env, UnityEnv, MujocoEnv }