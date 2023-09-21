# import
import asyncio
import os
import sys
import time
import copy
import math
from uuid import uuid4
from pathlib import Path
from pprint import pprint
import yaml

import numpy as np
import torch

sys.path.append('../..')
from kakiage.server import KakiageServerWSReceiveEvent, setup_server
from kakiage.tensor_serializer import serialize_tensors_to_bytes, deserialize_tensor_from_bytes

sys.path.append('./codes')
from models import make_net, get_io_shape
from server_utils import Server, ReplayBuffer
from training_utils import Trainings, Optimizer

# setup server to distribute javascript and communicate
kakiage_server = setup_server()
app = kakiage_server.app


def fix_seed(seed):    
    modules = ['random', 'numpy', 'pytorch']
    try:
        random.seed(seed)
    except:
        modules.remove('random')
    
    try:
        np.random.seed(seed)
    except:
        modules.remove('numpy')
    
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    except:
        modules.remove('pytorch')
    
    print(f'Seeds of {modules} are set to {seed}.')

      
class Arguments():
    def __init__(self):
        
        # args
        with open('codes/config/config.yaml') as f:
            cfg = yaml.safe_load(f)
        
        opt = dict(
            **cfg["experiment_config"], 
            **cfg["hyperparameters"]["training_options"], 
            **cfg["hyperparameters"]["testing_options"],
            **cfg["load_checkpoint"]
            )
        
        for key, value in opt.items():
            setattr(self, key, value)
            
        # to avoid bugs
        if self.save_weights:
            assert self.test_freq <= self.save_freq
            
        self.experiment_name = "MuJoCo-double-pendulum-00"
        if self.seed == "rand":
            self.seed = int(np.random.rand()*1000)
        
    def wandb_init(self, trials_id):
        
        import wandb
        init_config = {
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "architecture": self.model_name,
            "dataset": self.dataset_name,
            "max_epochs": self.train_step_num,
            
            # for grouping in wandb
            "env": self.env_key,
            "method": self.prioritized,
            "n_clients": f"L{self.n_learner_client_wait:02}A{self.n_actor_client_wait:02}",
            "trials_id": f"{trials_id:02}",
        }
        init_config["name"] = "/".join([init_config[key] for key in ["method", "n_clients", "trials_id"]])
        pprint(init_config)
        
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.experiment_name,
            
            # track hyperparameters and run metadata
            config=init_config,
            
            resume="allow",
            
            id = f"{trials_id:02}",
        )


async def wait_for_clients():
    global opt, sv
    """
    Wait for clients to join.
    """
    
    print(f"Waiting {opt.n_actor_client_wait + opt.n_learner_client_wait} clients to connect...")
    
    while len(sv.worker_ids) < opt.n_actor_client_wait + opt.n_learner_client_wait:
        
        # Record client id
        event = await sv.get_event()
        
        if isinstance(event, KakiageServerWSReceiveEvent):
            
            # Record first clients as learners
            if event.message['type']=='worker' and len(sv.init_learners) < int(opt.n_learner_client_wait):
                sv.init_learners.add(event.client_id)
                
            print(f"{len(sv.worker_ids)} / {opt.n_actor_client_wait + opt.n_learner_client_wait} clients connected")


async def collect_random_data():
    global opt, sv, replay_buffer
    """
    Let all workers to collect random sample as actors.
    
    1. Send messages to all workers.
    2. When workers connect back to server, then read uploaded data, add them to ReplayBuffer, and resend a message.
    """
    
    # Send messages
    for client_id in sv.worker_ids:
        await sv.twice(sv.send_msg_for_collecting_random_samples)(client_id)

    print('Start collecting init data in each actor.')
    
    while replay_buffer.size < int(opt.start_timesteps):
        event = await sv.get_event()
        
        if isinstance(event, KakiageServerWSReceiveEvent):
        
            # new worker
            if event.message['type']=='worker':
                await sv.twice(sv.send_msg_for_collecting_random_samples)(event.client_id)
            
            # known worker, finished collecting data
            elif event.message['type']=='actor':
                
                # Load data from blobs
                buffer_id = event.message['id']
                buffer_arrays_loaded = deserialize_tensor_from_bytes(kakiage_server.blobs[buffer_id])
                sv.buffer_ids_queue[event.client_id].remove(buffer_id)
                del kakiage_server.blobs[buffer_id]
                
                # Update ReplayBuffer
                replay_buffer.add(buffer_arrays_loaded)
                print(f"Get connection from client: {event.client_id}. Buffer size: {replay_buffer.size} / {opt.start_timesteps}")
                
                # Send message for next
                await sv.twice(sv.send_msg_for_collecting_random_samples)(event.client_id)
            
        else:
            print("unexpected event")

    print(f'\nRandom data collection finished.')
    print(f'Replay_buffer : size = {replay_buffer.size}, state.shape = {replay_buffer.state.shape}\n')


def upload_weights(weight_types):
    global sv
    """
    Generate weight ids and upload them to blob. Return ids.
    """
    
    for weight_type in weight_types:
        
        # Generate weight id
        weight_item_id = uuid4().hex
        sv.weight_item_ids[weight_type] = weight_item_id
        
        # Upload weight to blob
        kakiage_server.blobs[weight_item_id] = serialize_tensors_to_bytes(sv.weights[weight_type])
        
        # Delete old weight from blob
        weight_item_id_to_delete = sv.weight_item_ids_to_delete[weight_type].add(weight_item_id)
        if weight_item_id_to_delete is not None:
            del kakiage_server.blobs[weight_item_id_to_delete]


async def get_gradient():
    global opt, sv, replay_buffer
    """
    Let learners to compute gradients. Then, update global model weights via optimizer.
    
    1. Get data minibatch, which consists of pairs of [state, next_state, action, reward, done].
    2. Split minibatch to chunks of number of learners.
    3. Upload chunks and send messages to learners.
    4. Wait for learners to connect back. At the same time, also wait for other clients such as actors or visualizers.
    5. After collecting all local gradients from learners, compute global gradient and return it.
    """
    
    # Get minibatch
    indices, minibatch = replay_buffer.sample(opt.batch_size, sv.n_classes)
    
    # Give training data to learners
    sv.chunk_sizes = {}; sv.dataset_item_ids = {}; sv.grad_item_ids = {}; sv.td_item_ids = {};
    data_ind = dict()
    chunk_size = math.ceil(opt.batch_size / len(sv.learner_ids))
    for c, learner_id in enumerate(sv.learner_ids):
        
        # Record indices in ReplayBuffer of each data
        data_ind[learner_id] = indices[c*chunk_size:(c+1)*chunk_size]
        
        # Split data into chunks
        data_chunk = dict()
        data_size = set()
        for key, value in minibatch.items():
            data_chunk[key] = value[c*chunk_size:(c+1)*chunk_size]
            data_size.add(len(data_chunk[key]))
        
        # Add hyperparameters
        data_chunk["discount"] = np.array(opt.discount).astype(np.float32)
        
        # Record data size for each learners
        assert len(data_size) == 1, f'Data size must be same among all types of data. Got {", ".join(map(str, data_size))}.'
        sv.chunk_sizes[learner_id] = list(data_size)[0]

        # Upload chunks
        dataset_item_id = uuid4().hex
        kakiage_server.blobs[dataset_item_id] = serialize_tensors_to_bytes(data_chunk)
        sv.dataset_item_ids[learner_id] = dataset_item_id
        sv.learner_item_ids_to_delete.append(dataset_item_id)
        
        # Send msg to each actor
        await sv.twice(sv.send_msg_to_learner_for_collecting_grads)(learner_id, dataset_item_id)
            
    # Wait for all learners to complete
    complete_learner_grad_ids = {}
    connect_actor_ids = set()
    while len(complete_learner_grad_ids) < len(sv.learner_ids):
        event = await sv.get_event()
        
        # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 3
        step_elapsed_time = time.time() - step_start_time
        if step_elapsed_time > 20: # 20秒待つ
            print('\nERROR. BREAK! some of learners have got lost.\n')
            break
        # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 3 ここまで
        
        if isinstance(event, KakiageServerWSReceiveEvent):
            
            # TODO: 動的な割り当て
            # ここではあまり考えないようにしたいが...
            
            # new worker
            if event.message['type']=='worker':
                sv.actor_ids.add(event.client_id) # assign as actors
                await sv.twice(sv.send_msg_to_actor_for_collecting_samples)(event.client_id)
            
            # learners
            elif event.message['type']=='learner':
                complete_learner_grad_ids[event.client_id] = event.message['id']
                sv.working_ids[event.client_id] = time.time()
                print(f"[{len(complete_learner_grad_ids)}/{len(sv.learner_ids)}] Got connected by learner: {event.client_id}.")
                # TODO: まだ来ていないlearnerに割り当てられていたデータを配って計算させる？
                # TODO: 役割を終えたので重みを配ってactorとして働かせる？
                # TODO: データ配り直すのがめんどい
            
            # actors
            elif event.message['type']=='actor':
                buffer_id = event.message['id']
                sv.buffer_ids_queue[event.client_id].remove(buffer_id) # remove id from queue
                
                if event.message['success']:
                    
                    # Load data from blobs
                    buffer_arrays_loaded = deserialize_tensor_from_bytes(kakiage_server.blobs[buffer_id])
                    del kakiage_server.blobs[buffer_id]
                    
                    # Update ReplayBuffer
                    replay_buffer.add(buffer_arrays_loaded)
                    print(f"Got connected by actor : {event.client_id} | Buffer size: {replay_buffer.size} / {replay_buffer.max_size} | reward: {buffer_arrays_loaded['reward'].sum()}")
                
                # reset and resend
                # 最初のランダム探索結果を遅れて受信したとき、learnerに割り当てられたclientにはactor用のメッセージは送らない
                if event.client_id in sv.actor_ids:
                    await sv.twice(sv.send_msg_to_actor_for_collecting_samples)(event.client_id)
                    connect_actor_ids.add(event.client_id)
                    sv.working_ids[event.client_id] = time.time()
            
            # testers
            elif event.message['type']=='tester':
                print("Got connected by tester, ignored")
                
            # visualizers
            elif event.message['type']=='visualizer':
                print('Got connected by Visualizer')
                await sv.send_weight_msg_to_visualizer(event.client_id)
            
        else:
            print(f"unexpected event: {event}")

            # No support for disconnection and dynamic addition of clients (in this implementation, server waits for disconnected client forever)
            # To support, handle event such as KakiageServerWSConnectEvent
    
    # server side updates
    grad_arrays = {}
    # print(sv.chunk_sizes, sv.dataset_item_ids, sv.grad_item_ids, sv.td_item_ids)
    # print("complete_learner_grad_ids", complete_learner_grad_ids)

    for client_id in complete_learner_grad_ids.keys():
            
        # Update priorities of ReplayBuffer
        td_item_id = sv.td_item_ids[client_id]
        ind, priority = data_ind[client_id], deserialize_tensor_from_bytes(kakiage_server.blobs[td_item_id])['td_for_update']
        replay_buffer.update_priority(ind, priority)
        
        # Compute weighted average of gradients
        chunk_size, grad_item_id = sv.chunk_sizes[client_id], sv.grad_item_ids[client_id]
        chunk_weight = chunk_size / opt.batch_size
        try:
            chunk_grad_arrays = deserialize_tensor_from_bytes(kakiage_server.blobs[grad_item_id])
        except KeyError:
            print(f"\nid: {grad_item_id} has somehow disappeared. Goodbye!\n")
            continue
        for k, v in chunk_grad_arrays.items():
            if k in grad_arrays: 
                grad_arrays[k] += v * chunk_weight
            else: 
                grad_arrays[k] = v * chunk_weight
    
    return complete_learner_grad_ids, connect_actor_ids, grad_arrays


async def get_test_results():
    global opt, sv, replay_buffer
    """
    Let all workers to inference and get test results.
    
    1. Let learners collect test results, since learners are free from tasks.
    2. Let actors collect test results when actors connect to the server.
    3. Update test values after collecting enough test results.
    """

    # Learners as testers
    for client_id in list(sv.learner_ids):
        sv.learner_ids.remove(client_id)
        sv.tester_ids.add(client_id)
        await sv.send_msg_to_tester(client_id)
    
    # Wait for testers
    complete_test_reward = []
    while len(complete_test_reward) < opt.n_test_trials:
        event = await sv.get_event()
        
        if isinstance(event, KakiageServerWSReceiveEvent):
            
            # new workers
            if event.message['type']=='worker':
                sv.tester_ids.add(event.client_id)
                await sv.send_msg_to_tester(event.client_id)
                
            # testers
            elif event.message['type']=='tester':
                if event.message['success']:
                    complete_test_reward.append(float(event.message['reward']))
                await sv.send_msg_to_tester(event.client_id)
            
            # learners
            elif event.message['type']=='learner':
                sv.learner_ids.remove(event.client_id)
                sv.tester_ids.add(event.client_id)
                await sv.send_msg_to_tester(event.client_id)
                
                # for current implementation, learners are not suppose to connect in this section.
                print(f"ERROR: Got connected by learner: {event.client_id}.")
            
            # actors
            elif event.message['type']=='actor':
                buffer_id = event.message['id']
                sv.buffer_ids_queue[event.client_id].remove(buffer_id)
                
                if event.message['success']:
                    
                    # Load data from blobs
                    buffer_arrays_loaded = deserialize_tensor_from_bytes(kakiage_server.blobs[buffer_id])
                    del kakiage_server.blobs[buffer_id]
                    
                    # Update ReplayBuffer
                    replay_buffer.add(buffer_arrays_loaded)
                    print(f"Got connected by actor : {event.client_id} | Buffer size: {replay_buffer.size} / {replay_buffer.max_size} | reward: {buffer_arrays_loaded['reward'].sum()}")
                
                # reassign to testers
                if event.client_id in sv.actor_ids:
                    sv.actor_ids.remove(event.client_id)
                sv.tester_ids.add(event.client_id)
                await sv.send_msg_to_tester(event.client_id)
            
            # visualiser
            elif event.message['type']=='visualizer':
                print('Got connected by Visualizer')
                await sv.send_weight_msg_to_visualizer(event.client_id)
            
        else:
            # TODO: 切断対応: disconnect
            print(f"unexpected event: {event}")
            
    return complete_test_reward

async def main():
    global opt, sv, replay_buffer
    
    # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 1
    global step_start_time
    # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 1 ここまで
    
    print("Maze2D reinforcement learning data parallel training sample")
    
    # args
    opt = Arguments()
    if opt.use_wandb:
        import wandb
    
    for trials_id in range(opt.n_trials):
        
        # skip previous trials
        if opt.continue_train:
            if trials_id < opt.start_trial_id:
                continue
        # ----- 何度も手動でページをリフレッシュするのが面倒なので暫定的な実装でお茶を濁す 1
            elif trials_id > opt.start_trial_id:
                previous_worker_ids = sv.worker_ids
        else:
            if trials_id > 0:
                previous_worker_ids = sv.worker_ids
        # ----- 何度も手動でページをリフレッシュするのが面倒なので暫定的な実装でお茶を濁す 1 ここまで
        
        # initial settings
        fix_seed(opt.seed + trials_id)
        sv = Server(opt, kakiage_server)
        training = Trainings()
        if opt.continue_train and trials_id==opt.start_trial_id:
            training.iter = opt.start_iter
        
        # Make neural network model
        print(f"Model: {opt.model_name}, dataset: {opt.dataset_name}, batch size: {opt.batch_size}")
        sv.input_dim, sv.n_classes = get_io_shape(opt.dataset_name)
        model = make_net(opt.model_name, sv.input_dim, sv.n_classes)
        
        # Initialize Replay Buffer
        replay_buffer = ReplayBuffer(opt=opt, state_dim=sv.input_dim, action_dim=1)
        
        # set dir name for save
        root_dir = Path(opt.save_weights_root_dir)/opt.experiment_name/f"{opt.env_key}_{opt.prioritized}_L{opt.n_learner_client_wait:02}A{opt.n_actor_client_wait:02}"/f"{trials_id:02}"
        if opt.continue_train and trials_id==opt.start_trial_id:
            replay_buffer = sv.load_buffers(root_dir/f"replay_buffer")
            
        # use wandb for visualization
        if opt.use_wandb:
            opt.wandb_init(trials_id, )
        
        # ----- 何度も手動でページをリフレッシュするのが面倒なので暫定的な実装でお茶を濁す 2
        if opt.continue_train:
            if trials_id > opt.start_trial_id:
                for client_id in previous_worker_ids:
                    await sv.twice(sv.send_msg_reload)(client_id)
        else:
            if trials_id > 0:
                for client_id in previous_worker_ids:
                    await sv.twice(sv.send_msg_reload)(client_id)
        # ----- 何度も手動でページをリフレッシュするのが面倒なので暫定的な実装でお茶を濁す 2 ここまで
        
        # Wait for clients to join
        await wait_for_clients()
        
        # clock
        if opt.continue_train and trials_id==opt.start_trial_id:
            start_time = float(str(list(root_dir.glob("start_*"))[0]).split("_")[-1])
        else:
            if opt.save_weights:
                start_time = time.time()
                root_dir.mkdir(parents=True, exist_ok=True)
                with open(root_dir/f"start_{start_time}", "w"):
                    pass
                
        # ここに、デバイスのパラメータを推定するコードを入れる！！！！！！！！
        

        # Collect random data
        await collect_random_data()
        
        # initial assignment # important
        for client_id in sv.worker_ids:
            if client_id in sv.init_learners:
                sv.learner_ids.add(client_id)
            else:
                sv.actor_ids.add(client_id)
                
        # Read weights and upload them to blob
        sv.weights['global'] = training.get_weights(model)
        sv.weights['target'] = copy.deepcopy(sv.weights['global'])
        if opt.continue_train and trials_id==opt.start_trial_id:
            save_path = sorted(list(root_dir.glob(f"*{opt.start_iter}*.pkl")))[-1]
            sv.load_weights(save_path)
        upload_weights(['global', 'target'])
        
        # Define optimizer
        optimizer = Optimizer(opt, sv.weights['global'])
        
        # Send message to actors and start collecting data
        for actor_id in sv.actor_ids:
            await sv.twice(sv.send_msg_to_actor_for_collecting_samples)(actor_id)
            
        # training main loop
        while training.iter < opt.train_step_num:
            print(f"Iter: {training.iter}")
            training.iter += 1
            
            # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 2
            step_start_time = time.time()
            # learners_all_alive = True
            # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 2 ここまで
            
            with torch.no_grad():
                
                # training
                # for i in range(opt.update_freq):
                # training.total_it += 1

                # Update weights
                try:
                    print('start to compute gradient')
                    connecting_learners, connecting_actors, grad = await get_gradient()
                except KeyError: # 絶妙なタイミングでlearnerが死ぬと、ここでエラーになるので例外処理する
                    print("KeyError in 'get_gradient()'")
                    connecting_learners = {}
                
                # Remove offline clients
                timeout_ids_to_remove = []
                for client_id, last_connect in sv.working_ids.items():
                    if time.time() - last_connect > 30:
                        timeout_ids_to_remove.append(client_id)
                for client_id in timeout_ids_to_remove:
                    del sv.working_ids[client_id]
                    
                # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 5
                if len(connecting_learners) == 0:
                    print("ERROR: All learners are disconnected. Reload.")
                    
                    training.iter -= 1
                    
                    print("sv.worker_ids")
                    pprint(sv.worker_ids)
                    print("sv.client_ids")
                    pprint(sv.client_ids)
                    print("sv.learner_ids")
                    pprint(sv.learner_ids)
                    print("sv.actor_ids")
                    pprint(sv.actor_ids)
                    
                    # previous_worker_ids = sv.worker_ids
                    previous_worker_ids = list(sv.working_ids.keys())
                    sv.reset_clients()
                    
                    print("previous_worker_ids")
                    pprint(previous_worker_ids)
                    
                    for client_id in previous_worker_ids:
                        await sv.twice(sv.send_msg_reload)(client_id)
                    
                    while len(sv.worker_ids) < opt.n_actor_client_wait + opt.n_learner_client_wait:
                        event = await sv.get_event()
                        if isinstance(event, KakiageServerWSReceiveEvent):
                            if event.message['type']=='worker':
                                if len(sv.init_learners) < int(opt.n_learner_client_wait):
                                    sv.init_learners.add(event.client_id)
                                print(f"{len(sv.worker_ids)} / {opt.n_actor_client_wait + opt.n_learner_client_wait} clients connected")
                            
                    for client_id in sv.worker_ids:
                        if client_id in sv.init_learners:
                            sv.learner_ids.add(client_id)
                        else:
                            sv.actor_ids.add(client_id)
                            await sv.twice(sv.send_msg_to_actor_for_collecting_samples)(client_id)
                    
                    continue
                # ----- 低確率でlearnerが死ぬバグがあるが、理由がわからないので適当に対処 5 ここまで
                optimizer.step(grad)
                
                # Remove used ids except weight ids
                # weight ids are removed in function "upload_weights"
                for item_id in sv.learner_item_ids_to_delete:
                    if item_id in kakiage_server.blobs.keys(): # bug? なんか勝手に消えたり残ったりする
                        del kakiage_server.blobs[item_id]
                sv.learner_item_ids_to_delete = []

                # Update target weights
                if training.iter % opt.target_update == 0:
                    sv.weights['target'] = copy.deepcopy(sv.weights['global'])
                
                # Upload global and target weights to blob
                upload_weights(['global', 'target'])
                
                # reassignment
                if len(connecting_learners) < len(sv.init_learners):
                    # 足りない分は、actorから補充する # なくなったら足していくだけやから
                    missing_learners = sv.init_learners - set(connecting_learners.keys())
                    print(f"missing learners: {missing_learners}")
                    for client_id in missing_learners:
                        sv.init_learners.remove(client_id)
                        sv.learner_ids.remove(client_id)
                    for i, client_id in enumerate(connecting_actors):
                        if i < len(missing_learners):
                            print(f"reassignment: new learner is '{client_id}'")
                            sv.init_learners.add(client_id)
                            sv.learner_ids.add(client_id)
                            sv.actor_ids.remove(client_id)
                
                # testing and reassignment
                if training.iter == 1 or training.iter % opt.test_freq == 0 or training.iter == opt.train_step_num:
                    if opt.save_weights or opt.continue_train:
                        elapsed_time = time.time() - start_time
                    
                    # Get test results. Here, only collecting rewards. 
                    save_reward = await get_test_results()
                    
                    # update test plot
                    stats = training.update_test_log(replay_buffer.size, save_reward, elapsed_time)
                    last_stats = {key: value[-1] for key, value in stats.items()}
                    if opt.use_wandb:
                        wandb.log(last_stats)
                    
                    # reassignment # important
                    
                    print('\ndebug')
                    print("sv.worker_ids")
                    pprint(sv.worker_ids)
                    print("sv.client_ids")
                    pprint(sv.client_ids)
                    print("sv.tester_ids")
                    pprint(sv.tester_ids)
                    print("sv.init_learners")
                    pprint(sv.init_learners)
                    print("sv.actor_ids")
                    pprint(sv.actor_ids)
                    
                    for client_id in list(sv.tester_ids):
                        if client_id in sv.init_learners:
                            sv.learner_ids.add(client_id)
                        else:
                            sv.actor_ids.add(client_id)
                            await sv.twice(sv.send_msg_to_actor_for_collecting_samples)(client_id)
                                    
                                
                    
                    print("sv.learner_ids")
                    pprint(sv.learner_ids)
                    print("sv.actor_ids")
                    pprint(sv.actor_ids)
                    
                    sv.tester_ids = set()
            
                if opt.save_weights:
                    if training.iter == 1 or training.iter % opt.save_freq == 0 or training.iter == opt.train_step_num:
                        
                        # save global weight
                        save_path = root_dir/f"{training.iter:08}iter_{last_stats['reward']:.3f}reward"
                        sv.save_weights(save_path, sv.weights['global'])
                        
                        # save replay buffer
                        save_path = root_dir/f"replay_buffer"
                        sv.save_buffers(save_path, replay_buffer)
                        
        if opt.use_wandb:
            wandb.finish()
        
    print("training ended. You can close the program by Ctrl-C.")
    # TODO: exit program (sys.exit(0) emits long error message)
    sys.exit(0)


asyncio.get_running_loop().create_task(main())
