import os
import glob
import torch
import numpy as np
import subprocess
from natsort import natsorted
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
        self.n_tasks = len(gif_dirs) - len(gif_dirs)%num_batch_tasks
        print("n_tasks:", self.n_tasks)

        self.scale, self.bias = load_scale_and_bias(state_path)
        self.scale = torch.from_numpy(np.array(self.scale, np.float32))
        self.bias = torch.from_numpy(np.array(self.bias, np.float32))

        from tqdm import tqdm
        
        tasks = []

        all_states = []
        for idx in tqdm(range(self.n_tasks)):
            demos = []
            for j in range(6,18):
                path = os.path.join(demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt")
                demos.append(torch.load(path)) # n,100,125,125,3
            visions, states, actions = [], [], []
            for demo in demos:
                visions.append(demo['vision'].to('cuda'))
                state = torch.matmul(demo['state'], self.scale) + self.bias
                states.append(state.to('cuda'))
                actions.append(demo['action'].to('cuda'))
                all_states.append(state)
            visions = torch.stack(visions)
            states = torch.stack(states)
            actions = torch.stack(actions)
            tasks.append({'vision':visions, 'state':states, 'action':actions})

        _cmd = "nvidia-smi"
        subprocess.call(_cmd.split())

        all_states = torch.stack(all_states)
        print(visions.shape, states.shape, actions.shape)
        print(torch.mean(all_states, [0,1]))
        print(torch.std(all_states, [0,1]))

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):
        return 0


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--demo_dir', type=str, default='sim_push/')
    parser.add_argument('--state_path', type=str, default="scale_and_bias_sim_push.pkl")
    parser.add_argument('--num_batch_tasks', type=int, default=64)
    parser.add_argument('--train_n_shot', type=int, default=1)
    parser.add_argument('--test_n_shot', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    train_task = MILTaskset(demo_dir=args.demo_dir, 
                            state_path=args.state_path, 
                            num_batch_tasks=args.num_batch_tasks,
                            train_n_shot=args.train_n_shot, 
                            test_n_shot=1, 
                            val_size=0.1)
