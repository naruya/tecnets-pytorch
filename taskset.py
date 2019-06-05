import os
import glob
import torch
import numpy as np
from natsort import natsorted
from torch_utils import Taskset
from utils import make_cache


class MILTaskset(Taskset):
    def __init__(self,
                 demo_dir="../mil/data/sim_push/", train_n_shot=1, test_n_shot=1, mode='train', n_valid=76):

        self.mode = mode
        self.n_valid = n_valid
        self.demo_dir = demo_dir
        self.train_n_shot = train_n_shot
        self.test_n_shot = test_n_shot

        if os.path.exists(os.path.join(demo_dir, "cache")):
            print("cache exists")
        else:
            make_cache(demo_dir)

        # gif & pkl for all tasks
        gif_dirs = natsorted(glob.glob(os.path.join(demo_dir, "object_*")))
        pkl_files = natsorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
        assert len(gif_dirs) == len(pkl_files), ""
        self.n_tasks = len(gif_dirs)

        if mode == 'train':
            self.n_train = self.n_tasks - self.n_valid
            self.n_tasks = self.n_train
        elif mode == 'valid':
            self.n_train = self.n_tasks - self.n_valid
            self.n_tasks = self.n_valid
        elif mode == 'test':
            pass
        else:
            assert False, ""

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):

        if self.mode == 'valid':
            idx += self.n_train

        # TODO: とりあえず今は1-shotの場合しか想定していないので，増やすとおかしくなる
        train_indices = np.random.choice(range(0,6), size=self.train_n_shot, replace=False)
        test_indices = np.random.choice(range(18,24), size=self.test_n_shot, replace=False)

        # n,100,125,125,125,3
        train_demos = [torch.load(os.path.join(self.demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt")) for j in train_indices]
        test_demos = [torch.load(os.path.join(self.demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt")) for j in test_indices]

        return {
            "train": {
                "vision": ((torch.stack([demo['vision'] for demo in train_demos]).permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5),
                "state": torch.stack([demo['state'] for demo in train_demos]),
                "action": torch.stack([demo['action'] for demo in train_demos])
            },
            "test": {
                "vision": ((torch.stack([demo['vision'] for demo in test_demos]).permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5),
                "state": torch.stack([demo['state'] for demo in test_demos]),
                "action": torch.stack([demo['action'] for demo in test_demos])
            }, 
            'task_idx': idx
        }
