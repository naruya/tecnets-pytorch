import os
import glob
import torch
import numpy as np
import subprocess
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import Dataset as Taskset
from utils import make_cache
from utils import load_scale_and_bias


class MILTaskset(Taskset):
    def __init__(self,
                 demo_dir="../mil/data/sim_push/", state_path=None, num_batch_tasks=1,
                 train_n_shot=1, test_n_shot=1, valid=False, val_size=None):

        self.demo_dir = demo_dir
        self.train_n_shot = train_n_shot
        self.test_n_shot = test_n_shot
        self.valid = valid
        self.val_size = val_size

        if os.path.exists(os.path.join(demo_dir, "cache")):
            print("cache exists")
        else:
            make_cache(demo_dir)

        # gif & pkl for all tasks
        gif_dirs = natsorted(glob.glob(os.path.join(demo_dir, "object_*")))
        pkl_files = natsorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
        self.n_tasks = len(gif_dirs) - len(gif_dirs)%num_batch_tasks # drop last
#         self.n_tasks = 128 # len(gif_dirs) - len(gif_dirs)%num_batch_tasks # drop last

        if val_size:
            self.n_valid = val_size
            self.n_train = self.n_tasks - self.n_valid
            if not valid:
                self.n_tasks = self.n_train
            else:
                self.n_tasks = self.n_valid

        self.scale, self.bias = load_scale_and_bias(state_path)
        self.scale = torch.from_numpy(np.array(self.scale, np.float32))
        self.bias = torch.from_numpy(np.array(self.bias, np.float32))

        print("n_tasks:", self.n_tasks)
        print("load all tasks")
        self.tasks = []

        for idx in tqdm(range(self.n_tasks)):
            if valid:
                idx += self.n_train
            demos = []
            for j in range(6,18): # [1]
                path = os.path.join(demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt")
                demos.append(torch.load(path)) # {vision, state, action}
            visions, states, actions = [], [], []
            for demo in demos:
                visions.append(demo['vision'].to('cuda')) # Don't normalize now!
                state = torch.matmul(demo['state'], self.scale) + self.bias
                states.append(state.to('cuda'))
                actions.append(demo['action'].to('cuda')) # -2.0~+2.0
            visions = torch.stack(visions) # 12,100,64,64,3
            states = torch.stack(states)   # 12,100,20
            actions = torch.stack(actions) # 12,100,7
            self.tasks.append({'vision':visions, 'state':states, 'action':actions})

        _cmd = "nvidia-smi"
        subprocess.call(_cmd.split())

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):

        train_indices = np.random.choice(range(12), size=self.train_n_shot, replace=False)
        test_indices = np.random.choice(list(set(range(12))-set(train_indices)),
                                        size=self.test_n_shot, replace=False)

        task = self.tasks[idx]

        train_vision = task['vision'][train_indices] # k,100,64,64,3
        train_state = task['state'][train_indices]   # k,100,20
        train_action = task['action'][train_indices] # k,100,7
        test_vision = task['vision'][test_indices]   # k,100,64,64,3
        test_state = task['state'][test_indices]     # k,100,20
        test_action = task['action'][test_indices]   # k,100,7

        train_vision = (train_vision.permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5
        test_vision = (test_vision.permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5

        return {
            "train-vision": torch.unsqueeze(train_vision, 0),
            "train-state": torch.unsqueeze(train_state, 0),
            "train-action": torch.unsqueeze(train_action, 0),
            "test-vision": torch.unsqueeze(test_vision, 0),
            "test-state": torch.unsqueeze(test_state, 0),
            "test-action": torch.unsqueeze(test_action, 0),
        }

"""

[1] See issue #?

"""