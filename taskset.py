import os
import glob
import torch
import numpy as np
from natsort import natsorted
from torch_utils import Taskset
from utils import make_cache
from utils import load_scale_and_bias


class MILTaskset(Taskset):
    def __init__(self,
                 demo_dir="../mil/data/sim_push/", state_path=None,
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
        assert len(gif_dirs) == len(pkl_files), ""
        self.n_tasks = len(gif_dirs)

        if val_size:
            self.n_valid = int(self.n_tasks*val_size)
            self.n_train = self.n_tasks - self.n_valid
            if not valid:
                self.n_tasks = self.n_train
            else:
                self.n_tasks = self.n_valid

        self.scale, self.bias = load_scale_and_bias(state_path)
        self.scale = torch.from_numpy(np.array(self.scale, np.float32))
        self.bias = torch.from_numpy(np.array(self.bias, np.float32))

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):

        if self.valid:
            idx += self.n_train

        train_indices = np.random.choice(range(0,6), size=self.train_n_shot, replace=False)
        test_indices = np.random.choice(range(18,24), size=self.test_n_shot, replace=False)

        train_demos = [torch.load(os.path.join(self.demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt"))
                       for j in train_indices] # n,100,125,125,125,3
        test_demos = [torch.load(os.path.join(self.demo_dir, "cache", "task"+str(idx), "demo"+str(j)+".pt"))
                      for j in test_indices] # n,100,125,125,125,3

        return {
            "train": {
                "vision": ((torch.stack(
                    [demo['vision'] for demo in train_demos]).permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5),
                "state": torch.stack([
                    torch.matmul(demo['state'], self.scale) + self.bias for demo in train_demos]),
                "action": torch.stack([demo['action'] for demo in train_demos])
            },
            "test": {
                "vision": ((torch.stack([
                    demo['vision'] for demo in test_demos]).permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5),
                "state": torch.stack([
                    torch.matmul(demo['state'], self.scale) + self.bias for demo in test_demos]),
                "action": torch.stack([demo['action'] for demo in test_demos])
            }, 
            'task_idx': idx
        }
