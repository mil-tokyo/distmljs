
const sleep = (ms: number): Promise<null> => new Promise(resolve => setTimeout(resolve, ms))
const waitForClick = (msg: string): Promise<unknown> => {
    return new Promise(resolve => document.body.addEventListener("click", resolve))
};
const waitUntil = (flagfunc: () => boolean): Promise<null> => {
    return new Promise(resolve => {
        function checkFlag() {
            if (flagfunc()) {
                resolve(null);
            } else {
                setTimeout(checkFlag, 1);
            }
        }
        checkFlag();
    });
}

export { sleep, waitForClick, waitUntil }