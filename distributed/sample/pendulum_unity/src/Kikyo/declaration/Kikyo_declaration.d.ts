import { UnityInstance, UnityArguments, KikyoGlobal } from "../source/Kikyo_interface.js";

declare global{
    var Kikyo:KikyoGlobal;
    function createUnityInstance(
        canvasHtmlElement: HTMLCanvasElement,
        arguments: UnityArguments,
        onProgress?: (progression: number) => void
    ): Promise<UnityInstance>;
}