
const waitForSecond = (sec: number): Promise<null> => new Promise(resolve => setTimeout(resolve, sec*1000.0))
const waitForClick = (msg: string): Promise<unknown> => {
    console.log('wait for click...: ' + msg)
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

export { waitForSecond, waitForClick, waitUntil }