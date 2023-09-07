import { UnityInstance, UnityArguments } from "../source/Kikyo_interface.js";
import { KikyoGlobal } from "../source/Kikyo"

declare global{
    var Kikyo:KikyoGlobal;
    function createUnityInstance(
        canvasHtmlElement: HTMLCanvasElement,
        arguments: UnityArguments,
        onProgress?: (progression: number) => void
    ): Promise<UnityInstance>;
}