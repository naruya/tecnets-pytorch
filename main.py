import argparse
import os
import random
import time

import torch

from dataset import TecnetsDataset
from torch.utils.data import DataLoader
from tecnets import TecNets
from delogger.presets.profiler import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[3])
    parser.add_argument('--demo_dir', type=str, default='/root/datasets/mil_sim_push/')
    parser.add_argument('--num_batch_tasks', type=int, default=16)
    parser.add_argument('--num_load_tasks', type=int, default=16)
    parser.add_argument('--train_n_shot', type=int, default=1)
    parser.add_argument('--test_n_shot', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--emb_path', type=str, default=None)
    parser.add_argument('--ctr_path', type=str, default=None)
    parser.add_argument('--opt_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    assert args.num_batch_tasks % args.num_load_tasks == 0, ""

    meta_learner = TecNets(device=device, learning_rate=args.learning_rate)

    # for memory saving, for convenience, task_batch_size=1 (See tecnets*.py meta_train)
    train_task_loader = DataLoader(TecnetsDataset(demo_dir=args.demo_dir),
                                   batch_size=args.num_load_tasks,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)

    torch.backends.cudnn.benchmark = True

    meta_epochs = 4  # 400000/11
    start_time = time.time()
    for epoch in range(meta_epochs):
        print("# {}".format(epoch + 1))

        meta_learner.meta_train(task_loader=train_task_loader, num_batch_tasks=args.num_batch_tasks, num_load_tasks=args.num_load_tasks, epoch=epoch)
        epoch_time = time.time()

        print("each epoch cost ", (epoch_time - start_time) // (epoch + 1), " sec")

        if (epoch + 1) % 2 == 0:
            print("testing")
    
    total_time = time.time()
    print(f"{meta_epochs} step mean: ", (total_time - start_time) / meta_epochs)
    print(f"It will cost {(total_time - start_time) / meta_epochs * 400000 // (3600 * 24)} days for 400k step.")
    