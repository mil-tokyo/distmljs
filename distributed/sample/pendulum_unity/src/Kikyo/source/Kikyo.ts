import { Env } from "./Kikyo_Env";
import { KikyoGlobal, UnityInstance, UnityArguments } from "./Kikyo_interface";
import { waitUntil } from "./Kikyo_utility";

var Kikyo: KikyoGlobal = {
    callback: {},
    unityInstance: null,
    loaderReady: false,
    activeEnv: {},
    compress: true,

    createUnityInstance: async (): Promise<UnityInstance> => {
        console.log("Kikyo init called")

        const buildUrl = "Build";

        // create unity loader
        const loaderUrl = buildUrl + "/Build.loader.js";
        const loaderScript = document.createElement("script");
        loaderScript.src = loaderUrl;
        loaderScript.onload = async () => {
            Kikyo.loaderReady = true
        };
        document.body.appendChild(loaderScript);

        await waitUntil(() => { return Kikyo.loaderReady })

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
        console.log(canvas)

        let dataUrl = buildUrl + "/Build.data"
        let frameworkUrl = buildUrl + "/Build.framework.js"
        let codeUrl = buildUrl + "/Build.wasm"
        if (Kikyo.compress) {
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
                Kikyo.unityInstance = unityInstance;
                console.log(unityInstance)
            }).catch((message) => {
                alert(message);
            });

        await waitUntil(() => { return Kikyo.unityInstance != null })
        return Kikyo.unityInstance as UnityInstance;
    },

    getOrCreateUnityInstance: async (): Promise<UnityInstance> => {
        console.log("Kikyo getInstance called")

        if (Kikyo.unityInstance) {
            return Kikyo.unityInstance
        }
        else {
            return await Kikyo.createUnityInstance()
        }
    },

    getEnvironment: (name: string, index: number): Env => {
        const key = name + "_" + index.toString()
        if (key in Kikyo.activeEnv) {
            console.log(key)
            console.log(Kikyo.activeEnv)
            console.log(Kikyo.activeEnv[key])
            return Kikyo.activeEnv[key]
        } else {
            return new Env(name, index)
        }
    }
};

export { Kikyo }