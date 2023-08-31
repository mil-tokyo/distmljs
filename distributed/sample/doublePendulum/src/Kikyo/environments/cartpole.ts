import { Env, Observation } from "../exports";

export class CartPoleEnv extends Env {
  gravity: number;
  cart_mass: number;
  pole_mass: number;
  pole_length: number;
  force: number;

  x: number;
  x_dot: number;
  theta: number;
  theta_dot: number;

  max_episode_len: number;
  tau: number;

  x_threshold: number;
  theta_threshold: number;

  constructor() {
    super("cartpole", 1, 4)

    // cartpole
    this.gravity = 9.8;
    this.cart_mass = 1.0;
    this.pole_mass = 0.1;
    this.pole_length = 1.0;
    this.force = 10.0;

    // state
    this.x = 0;
    this.x_dot = 0;
    this.theta = 0;
    this.theta_dot = 0;

    // actor
    this.max_episode_len = 200
    this.tau = 0.01; // sec per frame

    // reward
    this.x_threshold = 2.4;
    this.theta_threshold = 0.418;
  }


  async _init(config?: object | undefined): Promise<Observation> {
    return await this._reset()
  }
  async _step(action: number[]): Promise<Observation> {
    // get force from action. if action==0, the direction of force is positive, and vice versa
    let force: number
    if (action[0] == 1) {
      force = this.force
    } else {
      force = - this.force
    }

    // do physics
    const cos_theta = Math.cos(this.theta);
    const sin_theta = Math.sin(this.theta);
    const total_mass = this.pole_mass + this.cart_mass;
    const pole_mass_length = this.pole_mass * this.pole_length / 2;

    const temp_acc = (
      force + pole_mass_length * (this.theta_dot ** 2) * sin_theta
    ) / total_mass
    const theta_acc = (this.gravity * sin_theta - cos_theta * temp_acc) / (
      this.pole_length / 2 * (4.0 / 3.0 - this.pole_mass * cos_theta ** 2 / total_mass)
    )
    const x_acc = temp_acc - pole_mass_length * theta_acc * cos_theta / total_mass
    // ↑ なんでこの式でうまくいくのかは分からなくなったけどがんばって求めた記憶がある

    // update state
    this.x += this.tau * this.x_dot
    this.x_dot += this.tau * x_acc
    this.theta += this.tau * this.theta_dot
    this.theta_dot += this.tau * theta_acc

    // get reward: always 1.0
    const reward = 1.0

    // done?
    let terminated: boolean = false;
    if (Math.abs(this.x) > this.x_threshold || Math.abs(this.theta) > this.theta_threshold) {
      terminated = true;
    }

    return { "state": [this.x, this.x_dot, this.theta, this.theta_dot], "reward_dict": { "constant": reward }, "terminated": terminated }
  }

  async _reset(): Promise<Observation> {
    this.x = (Math.random() * 2 - 1) * 0.05
    this.x_dot = (Math.random() * 2 - 1) * 0.05
    this.theta = (Math.random() * 2 - 1) * 0.05
    this.theta_dot = (Math.random() * 2 - 1) * 0.05
    return { "state": [this.x, this.x_dot, this.theta, this.theta_dot], "reward_dict": { "constant": 0 }, "terminated": false }
  }

}