import { mujoco } from "../declaration/mujoco_wasm";
import { Env } from "./Kikyo_Env";

interface UnityInstance {
    SendMessage(
        gameObjectName: string,
        methodName: string,
        argument: string
    ): void;
    SetFullscreen(fullScreen: boolean): void;
    Quit(): Promise<void>;
}

interface KikyoMudule {
    loaderReady: boolean,
    instance: any,
    createInstance: () => Promise<any>,
    getOrCreateInstance: () => Promise<any>,
}

interface KikyoMujocoMudule extends KikyoMudule{
    instance : mujoco | null
}

interface KikyoUnityMudule extends KikyoMudule{
    compress : boolean,
    instance : UnityInstance | null,
    createInstance: () => Promise<UnityInstance>,
    getOrCreateInstance: () => Promise<UnityInstance>,
}

interface UnityArguments {
    dataUrl: string,
    frameworkUrl: string,
    codeUrl: string,
    streamingAssetsUrl: string,
    companyName: string,
    productName: string,
    productVersion: string,
}

interface Observation {
    state: number[]
    reward_dict: { [name: string]: number };
    terminated: boolean;
}
interface SendValue {
    EnvName: string,
    Index?: number,
    Token: string,
    Action?: number[],
    Config?: object,
}
type KikyoUnityMethod = 'CreateEnvironment' | 'StepEnvironment' | 'ResetEnvironment'

export { KikyoMudule, UnityInstance, KikyoMujocoMudule, KikyoUnityMudule, KikyoUnityMethod, SendValue, Observation, UnityArguments }