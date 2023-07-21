
const sleep = (ms: number): Promise<null> => new Promise(resolve => setTimeout(resolve, ms))
const waitForClick = (msg: string): Promise<unknown> => {
    console.log(`wait for click ${msg}`)
    return new Promise(resolve => document.body.addEventListener("click", resolve))
};
const waitUntil = (flagfunc: () => boolean): Promise<null> => {
    console.log('waitUntil called')
    return new Promise(resolve => {
        function checkFlag() {
            if (flagfunc()) {
                console.log('waitUntil flg resolve')
                resolve(null);
            } else {
                setTimeout(checkFlag, 1);
            }
        }
        checkFlag();
    });
}

export {sleep,waitForClick,waitUntil}