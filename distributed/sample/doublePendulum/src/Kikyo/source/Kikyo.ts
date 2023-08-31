import { MujocoInstance } from "../declaration/mujoco_wasm";
import { UnityEnv, Env } from "./Kikyo_Env";
import { KikyoUnityMudule, KikyoMujocoMudule, UnityInstance, UnityArguments, KikyoUnityMethod } from "./Kikyo_interface";
import { waitUntil } from "./Kikyo_utility";

class KikyoGlobal {
    callback: { [token: string]: Function } = {};
    activeEnv: { [key: string]: Env } = {};
    unity: KikyoUnityMudule = {
        compress: false,
        instance: null,
        loaderReady: false,
        createInstance: async (): Promise<UnityInstance> => {
            const buildUrl = "Build";
            // create unity loader
            const loaderUrl = buildUrl + "/Build.loader.js";
            const loaderScript = document.createElement("script");
            loaderScript.src = loaderUrl;
            loaderScript.onload = async () => {
                Kikyo.unity.loaderReady = true
            };
            document.body.appendChild(loaderScript);

            await waitUntil(() => { return Kikyo.unity.loaderReady })

            //create or find canvas
            let canvas = document.querySelector("#unity-canvas") as HTMLCanvasElement;
            if (canvas == null) {
                console.log("create canvas");
                canvas = document.createElement("canvas");
                canvas.id = "unity-canvas"
                canvas.width = 960
                canvas.height = 600
                canvas.style.width = "960px";
                canvas.style.height = "600px";
                document.body.appendChild(canvas)
                await waitUntil(() => document.querySelector("#unity-canvas") != null)
            }

            let dataUrl = buildUrl + "/Build.data"
            let frameworkUrl = buildUrl + "/Build.framework.js"
            let codeUrl = buildUrl + "/Build.wasm"
            if (Kikyo.unity.compress) {
                dataUrl += ".gz"
                frameworkUrl += ".gz"
                codeUrl += ".gz"
            }

            //create config
            const config: UnityArguments = {
                dataUrl: dataUrl,
                frameworkUrl: frameworkUrl,
                codeUrl: codeUrl,
                streamingAssetsUrl: "StreamingAssets",
                companyName: "iykuetGames",
                productName: "KikyoRL",
                productVersion: "0.0.1",
            };

            // create unity instance
            createUnityInstance(canvas, config, (progress) => { })
                .then((unityInstance) => {
                    Kikyo.unity.instance = unityInstance;
                }).catch((message) => {
                    alert(message);
                });

            await waitUntil(() => { return Kikyo.unity.instance != null })

            return Kikyo.unity.instance as UnityInstance;

        },

        getOrCreateInstance: async (): Promise<UnityInstance> => {
            if (Kikyo.unity.instance) {
                return Kikyo.unity.instance
            }
            else {
                return await Kikyo.unity.createInstance()
            }
        },
    }
    mujoco: KikyoMujocoMudule = {
        loaderReady: false,
        instance: null,
        createInstance: async (): Promise<MujocoInstance> => {
            const buildUrl = "sources/mujoco";
            const loaderScript2 = document.createElement("script");
            loaderScript2.src = buildUrl + "/mujoco_loader.js";
            loaderScript2.type = 'module';
            loaderScript2.onload = async () => {
                await waitUntil(() => { return (window as any).mujoco_ready })
                const mujoco = (window as any).mujoco
                mujoco.FS.mkdir('/working');
                mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
                Kikyo.mujoco.loaderReady = true;
                Kikyo.mujoco.instance = mujoco as MujocoInstance;
                delete (window as any).mujoco;
            }
            document.body.appendChild(loaderScript2);

            await waitUntil(() => { return Kikyo.mujoco.loaderReady })
            await waitUntil(() => { return Kikyo.mujoco.instance != null })

            console.log('mujoco instance created !!')

            return Kikyo.mujoco.instance as MujocoInstance;
        },
        getOrCreateInstance: async (): Promise<MujocoInstance> => {
            if (Kikyo.mujoco.instance) {
                return Kikyo.mujoco.instance
            }
            else {
                return await Kikyo.mujoco.createInstance()
            }
        },
    }
};

window.Kikyo = window.Kikyo ?? new KikyoGlobal()

export { KikyoGlobal }