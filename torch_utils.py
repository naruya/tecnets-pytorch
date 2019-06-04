import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Pool


class Taskset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TaskLoader(object):
    def __init__(self, taskset, batch_size=1, shuffle=True):
        self.taskset = taskset
        self.batch_size = batch_size
        # idx iterator
        self.reset()

    def __iter__(self):
        return self

    def get_task(self, idx):
        return self.taskset[idx]

    def __next__(self):
        # batch_task = [self.taskset[next(self.sample_iter)] for _ in range(self.batch_size)]
        # ここの並列化はそんなに効果ななかった(-1にしているとシングルプロセスよりはやい時もあった)
        # backend default:loky, backend='threading'でマルチスレッドにできる
        batch_task = Parallel(n_jobs=1)([delayed(self.get_task)(next(self.sample_iter)) for _ in range(self.batch_size)])
        return batch_task

    def __len__(self):
        return len(self.taskset) // self.batch_size

    def reset(self):
        self.sample_iter = iter(
            np.random.permutation(np.arange(len(self.taskset))))
