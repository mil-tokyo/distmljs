
let Buffer = buffer.Buffer;

function downloadFile(url, outputPath) {
    return fetch(url)
        .then(x => x.arrayBuffer())
        .then(x => {
            FS.writeFile(outputPath, Buffer.from(x));
        });

}

var Module = {
    onRuntimeInitialized: function () {
        FS.mkdir('/working');
        FS.mount(MEMFS, { root: '.' }, '/working');
        downloadFile('sources/mujoco/simple.xml', '/working/simple.xml').then(() => {
            let model = Module.Model.load_from_xml("/working/simple.xml");
            let state = new Module.State(model);

            let simulation = new Module.Simulation(model, state);

            for (var i = 0; i < 100; i++) {
                simulation.step();
            }
        });
    }
};