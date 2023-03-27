export class Env {
  mass: number;
  radius: number;
  force_mag: number;
  tau: number;
  max_episode_len: number;
  maze: string[][];
  x: number;
  y: number;
  x_dot: number;
  y_dot: number;
  x_acc: number;
  y_acc: number;
  goal_x: number;
  goal_y: number;
  running_potential: number;
  collide: boolean;
  
  constructor() {
    // information of cart
    this.mass = 4.0; // 見つからなくて適当に決めてしまった
    this.radius = 0.2; // 見つからなくて適当に決めてしまった
    this.force_mag = 3.0; // 見つからなくて適当に決めてしまった
    this.tau = 0.2;

    // information of actor env
    this.max_episode_len = 200;

    this.maze = this.create_maze();
    this.x = -1;
    this.y = -1;
    this.x_dot = 0;
    this.y_dot = 0;
    this.x_acc = 0;
    this.y_acc = 0;
    this.goal_x = -1;
    this.goal_y = -1;

    this.running_potential = 0;
    this.collide = false;
  }

  create_maze (): string[][] {
    const maze_name = 'maze2d-umaze-dense';

    // const maze = [
    //   ['o', 'o', 'o'],
    // ];
    // const maze = [
    //   ['o', 'o', 'o'],
    //   ['o', 'x', 'o'],
    //   ['o', 'x', 'o'],
    // ];
    // const maze = [
    //   ['o', 'o', 'o', 'o'],
    //   ['o', 'x', 'x', 'o'],
    //   ['o', 'x', 'x', 'o'],
    //   ['o', 'o', 'o', 'o'],
    // ];
    // const maze = [
    //   ['o', 'o', 'x', 'o', 'o'],
    //   ['o', 'o', 'o', 'x', 'o'],
    //   ['x', 'x', 'o', 'o', 'o'],
    //   ['o', 'x', 'o', 'x', 'o'],
    //   ['o', 'o', 'o', 'o', 'o'],
    // ];
    const maze = [
      ['o', 'x', 'o', 'o', 'o', 'o', 'x'],
      ['o', 'x', 'x', 'o', 'x', 'o', 'o'],
      ['o', 'o', 'o', 'o', 'x', 'x', 'o'],
      ['x', 'o', 'x', 'x', 'x', 'o', 'o'],
      ['o', 'o', 'o', 'o', 'o', 'o', 'x'],
      ['o', 'x', 'x', 'x', 'x', 'o', 'o'],
      ['o', 'x', 'o', 'o', 'o', 'o', 'x'],
    ];
    // const maze = [
    //   ['o', 'o', 'o', 'x', 'x', 'o', 'x', 'x', 'o', 'o'],
    //   ['o', 'x', 'o', 'o', 'x', 'o', 'o', 'o', 'o', 'x'],
    //   ['o', 'x', 'x', 'o', 'x', 'x', 'x', 'o', 'x', 'x'],
    //   ['o', 'o', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
    //   ['x', 'o', 'o', 'o', 'x', 'o', 'x', 'x', 'x', 'o'],
    //   ['o', 'o', 'x', 'x', 'x', 'o', 'o', 'o', 'x', 'o'],
    //   ['o', 'x', 'x', 'o', 'o', 'o', 'x', 'o', 'o', 'o'],
    //   ['o', 'x', 'o', 'o', 'x', 'o', 'x', 'o', 'x', 'o'],
    //   ['o', 'x', 'o', 'x', 'x', 'o', 'x', 'x', 'x', 'o'],
    //   ['o', 'o', 'o', 'x', 'o', 'o', 'o', 'x', 'o', 'o'],
    // ];
    // const maze = [
    //   ['o', 'x', 'o', 'o', 'o', 'o', 'x', 'o', 'o', 'o', 'o', 'o', 'x', 'x', 'o'],
    //   ['o', 'x', 'x', 'o', 'x', 'o', 'o', 'o', 'x', 'o', 'x', 'o', 'o', 'o', 'o'],
    //   ['o', 'x', 'o', 'o', 'o', 'x', 'o', 'x', 'o', 'o', 'x', 'x', 'o', 'x', 'o'],
    //   ['o', 'o', 'x', 'x', 'o', 'o', 'o', 'x', 'o', 'x', 'o', 'x', 'o', 'x', 'o'],
    //   ['x', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'o', 'x', 'o', 'o', 'o', 'x', 'o'],
    //   ['o', 'o', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'o'],
    //   ['o', 'x', 'o', 'o', 'o', 'o', 'x', 'o', 'x', 'o', 'x', 'x', 'o', 'x', 'o'],
    //   ['o', 'x', 'o', 'x', 'x', 'o', 'x', 'o', 'x', 'o', 'x', 'x', 'o', 'o', 'o'],
    //   ['o', 'o', 'o', 'x', 'o', 'o', 'x', 'o', 'o', 'o', 'x', 'o', 'o', 'x', 'o'],
    //   ['x', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'x', 'x', 'o'],
    //   ['x', 'x', 'x', 'x', 'o', 'o', 'x', 'o', 'x', 'o', 'x', 'o', 'o', 'x', 'o'],
    //   ['o', 'x', 'o', 'x', 'x', 'o', 'x', 'o', 'o', 'o', 'x', 'x', 'o', 'x', 'o'],
    //   ['o', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'x', 'o', 'x', 'o', 'o', 'x', 'o'],
    //   ['o', 'x', 'x', 'o', 'o', 'x', 'x', 'o', 'o', 'o', 'x', 'x', 'o', 'x', 'x'],
    //   ['o', 'o', 'o', 'o', 'x', 'o', 'o', 'o', 'x', 'x', 'x', 'o', 'o', 'o', 'o'],
    // ];

    const h: number = maze.length;
    const w: number = maze[0].length;

    const edge: string[] = [...Array(w+2)].map((_, i) => 'x');
    let empty: string[][] = [edge]
    
    for (let row of maze) {
      let edgerow = ['x'].concat(row).concat(['x']);
      empty = empty.concat([edgerow]);
    }
    empty = empty.concat([edge]);

    return empty
  }

  reset (
    fix_pos: boolean,
    fix_goal: boolean
  ): number[] {

    const h: number = this.maze.length - 2;
    const w: number = this.maze[0].length - 2;
    let x: number, x_dot: number, y: number, y_dot: number, goal_x: number, goal_y: number

    // set init pos
    if (!fix_pos || (this.x===-1 && this.y===-1)) {
      while (true) {
        x = Math.random() * w + 1;
        y = Math.random() * h + 1;
        // for debug, fix spawn position
        x_dot = (Math.random() * 2 - 1) * 0.05; // max init x speed is 0.05
        y_dot = (Math.random() * 2 - 1) * 0.05; // max init y speed is 0.05
        
        if (this.maze[Math.floor(y)][Math.floor(x)] == 'o') {
          break;
        }
      }
      this.x = x;
      this.y = y;
      this.x_dot = x_dot;
      this.y_dot = y_dot;
    }

    // set goal pos
    if (!fix_goal || (this.goal_x===-1 && this.goal_y===-1)) {
      while (true) {
        goal_x = Math.random() * w + 1;
        goal_y = Math.random() * h + 1;
        // for debug, fix spawn position
        if (this.maze[Math.floor(goal_y)][Math.floor(goal_x)] == 'o') {
          break;
        }
      }
      this.goal_x = goal_x;
      this.goal_y = goal_y;
    }

    this.running_potential = 0;
    return [this.x, this.y, this.x_dot, this.y_dot, this.goal_x, this.goal_y]
  }

  step (
    [x, y, x_dot, y_dot]: [number, number, number, number],
    action: number
  ): [[number, number, number, number, number, number], number ,number]  {

    const h: number = this.maze.length - 2;
    const w: number = this.maze[0].length - 2;
    this.x = x;
    this.y = y;
    this.x_dot = x_dot;
    this.y_dot = y_dot;

    let x_force: number = 0, y_force:number = 0;
    // decide force based on given action
    if (action == 0) {
      x_force = this.force_mag;
      y_force = 0;
    } else if (action == 1) {
      x_force = - this.force_mag;
      y_force = 0;
    } else if (action == 2) {
      x_force = 0;
      y_force = this.force_mag;
    } else if (action == 3) {
      x_force = 0;
      y_force = -this.force_mag;
    } else {
      console.log('weird action detected')
    }
  
    // do physics
    // comment in bellow when use physics
    this.x_acc = x_force / this.mass;
    this.y_acc = y_force / this.mass;
    
    // move
    let x_next: number, y_next: number;
    let x_dif: number = this.tau * this.x_dot;
    let y_dif: number = this.tau * this.y_dot;
    this.collide = false
    
    // collision to walls
    let count = 0;
    while (true) {

      x_next = this.x + x_dif;
      y_next = this.y + y_dif;

      if (this.maze[Math.floor(y_next)][Math.floor(x_next)] == 'x') {
        this.collide = true
        x_dif = x_dif / 2;
        y_dif = y_dif / 2;
      } else {
        break;
      }

      if (count > 4) {
        x_dif = 0;
        y_dif = 0;
        break;
      }
      count++;
    }

    // update state
    this.x += x_dif;
    this.y += y_dif;

    if (!this.collide) {
      this.x_dot += this.tau * this.x_acc; // use physics
      this.y_dot += this.tau * this.y_acc; // use physics
      // this.x_dot = x_force; // use game like move
      // this.y_dot = y_force; // use game like move
    } else {
      if (Math.floor(this.x) != Math.floor(this.x + this.tau * this.x_dot)) {
        this.x_dot = 0;
      } else {
        this.x_dot += this.tau * this.x_acc; // use physics
        // this.x_dot = x_force; // use game like move
      }

      if (Math.floor(this.y) != Math.floor(this.y + this.tau * this.y_dot)) {
        this.y_dot = 0;
      } else {
        this.y_dot += this.tau * this.y_acc; // use physics
        // this.y_dot = y_force; // use game like move
      }
    }

    // get reward:
    const max_distance: number = Math.pow(Math.pow(w, 2) + Math.pow(h, 2), 0.5);
    const distance_to_goal: number = Math.pow(Math.pow(this.x - this.goal_x, 2) + Math.pow(this.y - this.goal_y, 2), 0.5);
    const distance_potential: number = 1 - distance_to_goal / max_distance;
    let reward_distance: number;
    if (this.running_potential < distance_potential) {
      reward_distance = 0.5;
    } else {
      reward_distance = -0.5;
    }
    this.running_potential = distance_potential;
    let reward_collision: number = 0;
    if (this.collide) {
      reward_collision = -0.30;
    }
    let constant_reward = -0.05;
    
    // done?
    let terminated: number = 0;
    let reward_reach: number = 0;
    if (distance_to_goal < this.radius * 2) {
      terminated = 1;
      reward_reach = 2;
    }

    const reward: number = constant_reward + reward_reach + reward_collision// + reward_distance;
    return [[this.x, this.y, this.x_dot, this.y_dot, this.goal_x, this.goal_y], reward, terminated]
  }

  normalize (
    [x, y, x_dot, y_dot, goal_x, goal_y]: number[]
  ): [number, number, number, number, number, number]  {

    const h: number = this.maze.length - 2;
    const w: number = this.maze[0].length - 2;
    const ret_x: number = (x - 1) / w - 0.5;
    const ret_y: number = (y - 1) / h - 0.5;
    const ret_goal_x: number = (goal_x - 1) / w - 0.5;
    const ret_goal_y: number = (goal_y - 1) / h - 0.5;
    const ret_x_dot: number = x_dot / w;
    const ret_y_dot: number = y_dot / h;

    return [ret_x, ret_y, ret_x_dot, ret_y_dot, ret_goal_x, ret_goal_y]
  }
}
