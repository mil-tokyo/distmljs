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

interface KikyoGlobal {
    callback: { [token: string]: Function },
    unityInstance: UnityInstance | null,
    loaderReady: boolean,
    activeEnv: { [key: string]: Env }
    createUnityInstance: () => Promise<UnityInstance>,
    getOrCreateUnityInstance: () => Promise<UnityInstance>,
    getEnvironment: (name: string, index: number) => Env,
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

export { KikyoGlobal, UnityInstance, KikyoUnityMethod, SendValue, Observation, UnityArguments }