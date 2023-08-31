import numpy as np
import copy
import pickle
from uuid import uuid4
import time
import websockets
from kakiage.server import KakiageServerWSConnectEvent, KakiageServerWSReceiveEvent


class Server():
    
    def __init__(self, opt, kakiage_server):
        self.opt = opt
        self.kakiage_server = kakiage_server
        
        # client ids
        self.reset_clients()
        
        # weights
        self.input_dim = None
        self.n_classes = None
        self.reset_weights()
        
        # data and grads
        self.chunk_sizes = []
        self.dataset_item_ids = []
        self.grad_item_ids = []
        self.td_item_ids = []
        
        # blob ids
        self.buffer_ids_queue = {} # type : dict # 何のためにあるのかあんまりわかってない
        self.learner_item_ids_to_delete = []
        
    def reset_clients(self):
        self.client_ids = set()
        self.worker_ids = set()
        self.client_role = {} # type: dict
        self.init_learners = set()
        self.learner_ids = set()
        self.actor_ids = set()
        self.tester_ids = set()
        self.visualizer_ids = set()
        self.working_ids = {} # type: dict
        
    def reset_weights(self):
        self.weights = {
            'global': None,
            'target': None
            }
        self.weight_item_ids_to_delete = {
            'global': self.weight_item_class(self.opt.weight_remain_epochs), 
            'target': self.weight_item_class(self.opt.weight_remain_epochs)
            }
        self.weight_item_ids = {
            'global': None, 
            'target': None
            }
        
    
    class weight_item_class():
        def __init__(self, max_len):
            self.ids = []
            self.max_len = max_len
        
        def add(self, weight_id):
            self.ids.append(weight_id)
            if len(self.ids) > self.max_len:
                id_to_remove = self.ids.pop(0)
                return id_to_remove
            else:
                return None
            
    def save_weights(self, save_path, weights):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path)+".pkl", "wb") as tf:
            pickle.dump(weights, tf)
        print(f"model is saved to: {str(save_path)}.pkl")
        
    def load_weights(self, save_path):
        with open(str(save_path), "rb") as tf:
            weight = pickle.load(tf)
        print(f"model is loaded from: {str(save_path)}")
        self.weights['global'] = copy.deepcopy(weight)
        self.weights['target'] = copy.deepcopy(weight)
        
    def save_buffers(self, save_path, replay_buffer):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path)+".pkl", "wb") as tf:
            pickle.dump(replay_buffer, tf)
        print(f"Replay Buffer is saved to: {str(save_path)}.pkl")
        
    def load_buffers(self, save_path):
        with open(str(save_path)+".pkl", "rb") as tf:
            replay_buffer = pickle.load(tf)
        print(f"replay_buffer is loaded from: {str(save_path)}.pkl")
        return replay_buffer
            
    # async processes
    
    def twice(self, async_func, sleep_time=5):
        # async_funcの引数の1つ目はclient_idじゃないといけない
        # TODO: キャッチするエラーの数を増やす
        async def try_3_times(client_id, *args):
            try:
                await async_func(client_id, *args)
            except (KeyError, websockets.exceptions.ConnectionClosedOK) as e:
                print(e)
                pass
                time.sleep(sleep_time)
                print(f"Failed to send msg to client: {client_id}. Try again.")
                try:
                    await async_func(client_id, *args)
                except (KeyError, websockets.exceptions.ConnectionClosedOK) as e:
                    print(e)
                    time.sleep(sleep_time)
                    print(f"Failed to send msg to client: {client_id}. Try again.")
                    try:
                        await async_func(client_id, *args)
                    except (KeyError, websockets.exceptions.ConnectionClosedOK) as e:
                        print(e)
                        pass
        return try_3_times
    
    async def get_event(self):
        while True:
            event = await self.kakiage_server.event_queue.get()
            if isinstance(event, KakiageServerWSConnectEvent):
                print(f"Got connected by client: {event.client_id}. ")
                self.client_ids.add(event.client_id)
                self.client_role[event.client_id] = 'unassigned'
            elif isinstance(event, KakiageServerWSReceiveEvent):
                self.client_role[event.client_id] = event.message['type']
                if event.message['type']=='worker':
                    self.buffer_ids_queue[event.client_id] = []
                    self.worker_ids.add(event.client_id)
                if event.message['type']=='visualizer':
                    if not event.client_id in self.visualizer_ids:
                            self.visualizer_ids.add(event.client_id)
            return event
    
    
    async def send_msg_reload(self, client_id):
        print(f"Send msg reload to: {client_id}")
        await self.kakiage_server.send_message(client_id, {
            "type": "reload",
        })
    
    
    async def send_msg_for_collecting_random_samples(self, client_id):
        print(f"Send random collection msg: {client_id}. ") 
        buffer_id = uuid4().hex
        self.buffer_ids_queue[client_id].append(buffer_id)
        await self.kakiage_server.send_message(client_id, {
            "client_id": client_id,
            "type": "actor",
            "env_key": self.opt.env_key,
            "random_sample": 1,
            "model": self.opt.model_name, # not in use when random sampling
            "inputShape": self.input_dim, # not in use when random sampling
            "nClasses": self.n_classes, # not in use when random sampling
            "buffer_id": buffer_id,
        })
        
        
    async def send_msg_to_actor_for_collecting_samples(self, client_id):
        buffer_id = uuid4().hex
        self.buffer_ids_queue[client_id].append(buffer_id)
        await self.kakiage_server.send_message(client_id, {
            "client_id": client_id,
            "type": "actor", 
            "env_key": self.opt.env_key,
            "random_sample": 0, 
            "model": self.opt.model_name, 
            "inputShape": self.input_dim, 
            "nClasses": self.n_classes, 
            "weight_actor_item_id": self.weight_item_ids['global'], 
            "buffer_id": buffer_id, 
        })
        
        
    async def send_msg_to_learner_for_collecting_grads(self, client_id, dataset_item_id):
        grad_item_id = uuid4().hex
        self.grad_item_ids[client_id] = grad_item_id
        self.learner_item_ids_to_delete.append(grad_item_id)
        td_item_id = uuid4().hex
        self.td_item_ids[client_id] = td_item_id
        self.learner_item_ids_to_delete.append(td_item_id)
        await self.kakiage_server.send_message(client_id, {
            "client_id": client_id,
            "type": "learner",
            "prioritized": self.opt.prioritized,
            "model": self.opt.model_name,
            "inputShape": self.input_dim,
            "nClasses": self.n_classes,
            "weight_learner": self.weight_item_ids['global'],
            "weight_target_learner": self.weight_item_ids['target'],
            "dataset": dataset_item_id,
            "grad": grad_item_id,
            "td": td_item_id,
        })
        
        
    async def send_msg_to_tester(self, client_id):
        print(f'send msg to tester: {client_id}')
        await self.kakiage_server.send_message(client_id, {
            "client_id": client_id,
            "type": "tester", 
            "env_key": self.opt.env_key,
            "model": self.opt.model_name, 
            "inputShape": self.input_dim, 
            "nClasses": self.n_classes, 
            "weight_item_id": self.weight_item_ids['global'], 
        })
        
        
    async def send_weight_msg_to_visualizer(self, client_id):
        print(f'send msg to visualizer: {client_id}')
        await self.kakiage_server.send_message(client_id, {
            "client_id": client_id,
            "type": "visualizer.weight",
            "env_key": self.opt.env_key,
            "model": self.opt.model_name, 
            "inputShape": self.input_dim, 
            "nClasses": self.n_classes, 
            "weight_actor_item_id": self.weight_item_ids['global'], 
        })
        
        
# replay buffer for asynchronous training
class ReplayBuffer(object):
    def __init__(self, opt, state_dim, action_dim, max_size=int(1e7)):
        self.opt = opt
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.val_names_dims = { # update here when you add new attribute
            "state": state_dim, 
            "action": action_dim, 
            "next_state": state_dim, 
            "reward": 1, 
            "not_done": 1,
        }
        self.clear()
        self.tree = SumTree(self.max_size)
        
        
    def clear(self):
        for name, dim in self.val_names_dims.items():
            setattr(self, name, np.zeros((self.max_size, dim), dtype=np.float32))


    def parse(self, loaded_buffer):
        ret = { # update here when add new attribute
            "state": loaded_buffer["state"],
            "action" : loaded_buffer["action"],
            "next_state": np.roll(loaded_buffer["state"], shift=-1, axis=0),
            "reward": loaded_buffer["reward"],
            "not_done": 1. - loaded_buffer["done"],
        }
        return ret
    
    
    def add(self, loaded_buffer):
        parsed_buffer = self.parse(loaded_buffer)
        
        # check if loaded buffer is consistent
        input_length = set([len(value) for value in parsed_buffer.values()])
        assert len(input_length) == 1, f'Buffer size must be same among attributes. Got {", ".join(map(str, input_length))}.'
        input_length = list(input_length)[0]
        assert input_length < self.max_size, f'Buffer size ({input_length}) must be smaller than max_buffer_size ({self.max_size}).'
        
        # update buffer data
        end_ptr = self.ptr + input_length
        slices = [slice(self.ptr, min(end_ptr, self.max_size)), slice(0, max(0, end_ptr - self.max_size))]
        for name in self.val_names_dims.keys():
            attr_array = getattr(self, name)
            for idx, src in zip(slices, np.split(parsed_buffer[name], [self.max_size - self.ptr])):
                attr_array[idx] = src
            
        # update priority
        ind, priority = np.concatenate([np.arange(self.max_size)[idx] for idx in slices]), loaded_buffer["td"].reshape(input_length)
        self.tree.batch_set(ind, priority)

        # update buffer info
        self.ptr = end_ptr % self.max_size
        self.size = min(self.size + input_length, self.max_size)


    def sample(self, batch_size, n_classes):
        
        if self.opt.prioritized == "rand":
            # random sampling
            ind = np.random.randint(0, self.size, size=batch_size)
        else:
            # weightened sampling
            ind = self.tree.sample(batch_size)
        
        # get data samples from buffer
        ret = dict([(name, getattr(self, name)[ind]) for name in self.val_names_dims.keys()])
        
        # すごく一時的な実装 # そのうちやめる
        ret["action_mask"] = np.eye(n_classes)[ret["action"].squeeze().astype(np.int64)].astype(np.float32)
        del ret["action"]

        return ind, ret
    
    
    def update_priority(self, ind, priority):
        self.tree.batch_set(ind, priority)
        
        
class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2


    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index


    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2