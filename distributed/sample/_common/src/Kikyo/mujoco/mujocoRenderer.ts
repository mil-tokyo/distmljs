import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { getPosition, getQuaternion, loadSceneFromURL, standardNormal, toMujocoPos } from './mujocoUtils';
import { Model, MujocoInstance, Simulation, State } from '../declaration/mujoco_wasm';

interface Meshes {
    cylinders: THREE.InstancedMesh,
    spheres: THREE.InstancedMesh,
}

class MujocoRenderer {
    params: { paused: boolean; help: boolean; ctrlnoiserate: number; ctrlnoisestd: number; keyframeNumber: number; };
    actuators: { [key: number]: number };
    mujoco_time: number;
    mujocoRoot: (THREE.Group & Meshes) | undefined;
    bodies: { [key: number]: THREE.Group };
    lights: { [key: number]: THREE.Light };
    tmpVec: any;
    tmpQuat: any;
    updateGUICallbacks: never[];
    container: HTMLDivElement;
    scene: THREE.Scene;
    camera: any;
    ambientLight: any;
    renderer: any;
    controls: any;
    // dragStateManager: any;
    model: Model | undefined;
    state: State | undefined;
    simulation: Simulation | undefined;

    constructor() {
        // Define Random State Variables
        this.params = { paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0 };
        this.actuators = {}
        this.mujoco_time = 0.0;
        this.bodies = {}, this.lights = {};
        this.tmpVec = new THREE.Vector3();
        this.tmpQuat = new THREE.Quaternion();
        this.updateGUICallbacks = [];

        this.container = document.createElement('div');
        document.body.appendChild(this.container);

        this.scene = new THREE.Scene();
        this.scene.name = 'scene';

        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
        this.camera.name = 'PerspectiveCamera';
        this.camera.position.set(2.0, 1.7, 1.7);
        this.scene.add(this.camera);

        this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
        this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
        this.ambientLight.name = 'AmbientLight';
        this.scene.add(this.ambientLight);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
        this.renderer.setAnimationLoop(this.render.bind(this));

        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 0.7, 0);
        this.controls.panSpeed = 2;
        this.controls.zoomSpeed = 1;
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.10;
        this.controls.screenSpacePanning = true;
        this.controls.update();

        window.addEventListener('resize', this.onWindowResize.bind(this));
    }


    async init(model: Model, state: State, simulation: Simulation, mujoco?: MujocoInstance) {
        console.log("init called")
        this.setModels(model, state, simulation);
        // Initialize the three.js Scene using the .xml Model in initialScene
        if (mujoco == undefined) {
            mujoco = Kikyo.mujoco.instance as MujocoInstance;
        }
        [this.model, this.state, this.simulation, this.bodies, this.lights, this.mujocoRoot] =
            await loadSceneFromURL(mujoco, this.model, this.state, this.simulation, this.scene);
        // this.renderer.setAnimationLoop(this.render.bind(this));

        console.log(`simulation timestep=${this.model?.getOptions().timestep}`)
    }

    async remove_models() {
        const rootObject = this.scene.getObjectByName("MuJoCo Root");
        if (rootObject) {
            this.scene.remove(rootObject);
        }
    }

    setModels(model: Model, state: State, simulation: Simulation) {
        this.model = model;
        this.state = state;
        this.simulation = simulation;
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    step(timeMS: number) {
        console.log(`render called: timeMS=${timeMS}`)

        if (this.model == undefined || this.state == undefined || this.simulation == undefined) {
            return;
        }

        this.controls.update();

        if (!this.params["paused"]) {
            console.log(`not paused: timestep=${this.model.getOptions().timestep} timeMS=${timeMS}, mujoco_time=${this.mujoco_time}`)
            let timestep = this.model.getOptions().timestep;
            if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
            while (this.mujoco_time < timeMS) {

                // Jitter the control state with gaussian random noise
                if (this.params["ctrlnoisestd"] > 0.0) {
                    let rate = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
                    let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
                    let currentCtrl = this.simulation.ctrl;
                    for (let i = 0; i < currentCtrl.length; i++) {
                        currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
                        this.actuators[i] = currentCtrl[i];
                    }
                }

                // Clear old perturbations, apply new ones.
                for (let i = 0; i < this.simulation.qfrc_applied.length; i++) { this.simulation.qfrc_applied[i] = 0.0; }
                // let dragged = this.dragStateManager.physicsObject;
                // if (dragged && dragged.bodyID) {
                //     for (let b = 0; b < this.model.nbody; b++) {
                //         if (this.bodies[b]) {
                //             getPosition(this.simulation.xpos, b, this.bodies[b].position);
                //             getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
                //             this.bodies[b].updateWorldMatrix(false, false);
                //         }
                //     }
                //     let bodyID = dragged.bodyID;
                //     this.dragStateManager.update(); // Update the world-space force origin
                //     let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * 250));
                //     let point = toMujocoPos(this.dragStateManager.worldHit.clone());
                //     this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);

                //     // TODO: Apply pose perturbations (mocap bodies only).
                // }

                this.simulation.step();

                this.mujoco_time += timestep * 1000.0;
            }

        } else if (this.params["paused"]) {
            console.log(`paused: timestep=${this.model.getOptions().timestep}`)
            // this.dragStateManager.update(); // Update the world-space force origin
            // let dragged = this.dragStateManager.physicsObject;
            // if (dragged && dragged.bodyID) {
            //     let b = dragged.bodyID;
            //     getPosition(this.simulation.xpos, b, this.tmpVec, false); // Get raw coordinate from MuJoCo
            //     getQuaternion(this.simulation.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

            //     let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
            //         .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
            //     if (this.model.body_mocapid[b] >= 0) {
            //         // Set the root body's mocap position...
            //         console.log("Trying to move mocap body", b);
            //         let addr = this.model.body_mocapid[b] * 3;
            //         let pos = this.simulation.mocap_pos;
            //         pos[addr + 0] += offset.x;
            //         pos[addr + 1] += offset.y;
            //         pos[addr + 2] += offset.z;
            //     } else {
            //         // Set the root body's position directly...
            //         let root = this.model.body_rootid[b];
            //         let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
            //         let pos = this.simulation.qpos;
            //         pos[addr + 0] += offset.x;
            //         pos[addr + 1] += offset.y;
            //         pos[addr + 2] += offset.z;
            //     }
            // }

            this.simulation.forward();
        }
    }

    render() {
        if (this.model == undefined || this.state == undefined || this.simulation == undefined) {
            return;
        }

        // Update body transforms.
        for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
                getPosition(this.simulation.xpos, b, this.bodies[b].position);
                getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
                this.bodies[b].updateWorldMatrix(false, false);
            }
        }

        // Update light transforms.
        for (let l = 0; l < this.model.nlight; l++) {
            if (this.lights[l]) {
                getPosition(this.simulation.light_xpos, l, this.lights[l].position);
                getPosition(this.simulation.light_xdir, l, this.tmpVec);
                this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
            }
        }

        // Update tendon transforms.
        let numWraps = 0;
        if (this.mujocoRoot && this.mujocoRoot.cylinders) {
            let mat = new THREE.Matrix4();
            for (let t = 0; t < this.model.ntendon; t++) {
                let startW = this.simulation.ten_wrapadr[t];
                let r = this.model.tendon_width[t];
                for (let w = startW; w < startW + this.simulation.ten_wrapnum[t] - 1; w++) {
                    let tendonStart = getPosition(this.simulation.wrap_xpos, w, new THREE.Vector3());
                    let tendonEnd = getPosition(this.simulation.wrap_xpos, w + 1, new THREE.Vector3());
                    let tendonAvg = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

                    let validStart = tendonStart.length() > 0.01;
                    let validEnd = tendonEnd.length() > 0.01;

                    if (validStart) { this.mujocoRoot.spheres.setMatrixAt(numWraps, mat.compose(tendonStart, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
                    if (validEnd) { this.mujocoRoot.spheres.setMatrixAt(numWraps + 1, mat.compose(tendonEnd, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
                    if (validStart && validEnd) {
                        mat.compose(tendonAvg, new THREE.Quaternion().setFromUnitVectors(
                            new THREE.Vector3(0, 1, 0), tendonEnd.clone().sub(tendonStart).normalize()),
                            new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r));
                        this.mujocoRoot.cylinders.setMatrixAt(numWraps, mat);
                        numWraps++;
                    }
                }
            }
            this.mujocoRoot.cylinders.count = numWraps;
            this.mujocoRoot.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
            this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
            this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
        }

        // Render!
        this.renderer.render(this.scene, this.camera);
    }
}

export { MujocoRenderer }