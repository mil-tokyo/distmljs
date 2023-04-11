export class Env {
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

  reset (
    fix_player: boolean, 
    fix_goal: boolean
  ): number[] {
    this.x = (Math.random() * 2 - 1) * 0.05
    this.x_dot = (Math.random() * 2 - 1) * 0.05
    this.theta = (Math.random() * 2 - 1) * 0.05
    this.theta_dot = (Math.random() * 2 - 1) * 0.05
    return [this.x, this.x_dot, this.theta , this.theta_dot]
  }

  step (
    [x, x_dot, theta, theta_dot]: [number, number, number, number],
    action: number
  ): [[number, number, number, number], number ,number]  {

    // get force from action. if action==0, the direction of force is positive, and vice versa
    let force: number
    if (action == 1) { 
      force = this.force
    } else {
      force = - this.force
    }
  
    // do physics
    const cos_theta = Math.cos(theta);
    const sin_theta = Math.sin(theta);
    const total_mass = this.pole_mass + this.cart_mass;
    const pole_mass_length = this.pole_mass * this.pole_length / 2;

    const temp_acc = (
      force + pole_mass_length * (theta_dot**2) * sin_theta
    ) / total_mass
    const theta_acc = (this.gravity * sin_theta - cos_theta * temp_acc) / (
      this.pole_length / 2 * (4.0 / 3.0 - this.pole_mass * cos_theta**2 / total_mass)
    )
    const x_acc = temp_acc - pole_mass_length * theta_acc * cos_theta / total_mass
    // ↑ なんでこの式でうまくいくのかは分からなくなったけどがんばって求めた記憶がある
    
    // update state
    x += this.tau * x_dot
    x_dot += this.tau * x_acc
    theta += this.tau * theta_dot
    theta_dot += this.tau * theta_acc
  
    // get reward: always 1.0
    const reward = 1.0
    
    // done?
    let terminated: number = 0;
    if (Math.abs(x) > this.x_threshold || Math.abs(theta) > this.theta_threshold) {
      terminated = 1;
    } 
    
    return [[x, x_dot, theta, theta_dot], reward, terminated]
  }
}