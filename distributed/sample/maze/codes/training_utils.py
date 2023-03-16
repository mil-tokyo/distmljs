import numpy as np

class Trainings():
    def __init__(self):
        # training
        self.iter = 0
        self.total_it = 0
        
        # visualization
        self.save_iters = []
        self.save_buffer_sizes = []
        self.save_reward = []
        self.save_elapsed_time = []
        
        
    def snake2camel(self, name):
        """
        running_mean -> runningMean
        PyTorch uses snake_case, kakiage uses camelCase
        """
        upper = False
        cs = []
        for c in name:
            if c == "_":
                upper = True
                continue
            if upper:
                c = c.upper()
            cs.append(c)
            upper = False
        return "".join(cs)
        
        
    # Extract weight values from model
    def get_weights(self, model):
        weights = {}
        for k, v in model.state_dict().items():
            # fc1.weight, fc1.bias, ...
            vnum = v.detach().numpy()
            if vnum.dtype == np.int64:
                vnum = vnum.astype(np.int32)
            weights[self.snake2camel(k)] = vnum
        return weights
    
    
    def update_test_log(self, size, reward, elapsed_time):
        reward = np.array(reward)
        self.save_iters.append(self.iter)
        self.save_buffer_sizes.append(size)
        self.save_reward.append(float(np.array(reward).mean()))
        self.save_elapsed_time.append(elapsed_time)
        stats = {
            'iters': self.save_iters, 
            'bfsize': self.save_buffer_sizes, 
            'reward': self.save_reward,
            'time': self.save_elapsed_time,
        }
        return stats


class Optimizer():
    def __init__(self, opt, weight):
        self.opt = opt
        self.weight = weight
        self.total_iter = 0
        if self.opt.adam:
            self.adam_initialize()
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            
            
    def is_trainable_key(self, name):
        if "running" in name:
            # runningMean, runningVar
            return False
        if "numBatchesTracked" in name:
            return False
        return True
        
        
    def adam_initialize(self):
        self.moment = {}
        self.rms = {}
        for k, v in self.weight.items():
            if self.is_trainable_key(k):
                self.moment[k] = np.zeros_like(v)
                self.rms[k] = np.zeros_like(v)
        
        
    def step(self, gradients):
        # self.wegihtはdictなので、keyで指定してvalueを変更するとsv.weights['global']のviewが変更され、値が更新される
        
        self.total_iter += 1
        for weight_k, (grad_k, grad_v) in zip(self.weight.keys(), gradients.items()):
            assert weight_k == grad_k
            key = weight_k
            if self.is_trainable_key(key):
                if self.opt.adam: # 参考: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                    self.moment[key] = self.moment[key] * self.beta1 + (1-self.beta1) * grad_v
                    self.rms[key] = self.rms[key] * self.beta2 + (1-self.beta2) * grad_v * grad_v
                    self.weight[key] -= self.opt.lr * (self.moment[key] / (1-self.beta1**self.total_iter)) / np.sqrt((self.rms[key] / (1-self.beta2**self.total_iter)) + self.epsilon)
                
                else: # SGD
                    self.weight[key] -= self.opt.lr * grad_v
            
            else:
                # not trainable = BN stats = average latest value
                self.weight[key][...] = grad_v
                