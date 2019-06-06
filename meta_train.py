# export CUDA_VISIBLE_DEVICES="2, 3"

import argparse
from datetime import datetime
import os
import random
from tensorboardX import SummaryWriter
import torch

from taskset import MILTaskset
from torch_utils import TaskLoader
from tecnets import TecNets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--demo_dir', type=str, default='../mil/data/sim_push/')
    parser.add_argument('--num_batch_tasks', type=int, default=64)
    parser.add_argument('--state_path', type=str, default="../mil/data/sim_push_common/scale_and_bias_sim_push.pkl")
    parser.add_argument('--train_n_shot', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    writer = SummaryWriter()
    log_dir = "./logs/" + datetime.now().strftime('%Y%m%d-%H%M%S')+"-B"+str(args.num_batch_tasks)\
                          +"-train_n_shot"+str(args.train_n_shot)
    print(log_dir)
    os.mkdir(log_dir)

    meta_learner = TecNets(device=device, log_dir=log_dir)

    train_task_loader = TaskLoader(MILTaskset(
        demo_dir=args.demo_dir, state_path=args.state_path,
        train_n_shot=args.train_n_shot, test_n_shot=1), batch_size=args.num_batch_tasks)
    # valid_task_loader = TaskLoader(MILTaskset(
    #     demo_dir=args.demo_dir, state_path=args.state_path,
    #     train_n_shot=1, test_n_shot=1, valid=True, val_size=0.1),batch_size=args.num_batch_tasks)

    meta_epochs = 585

    for epoch in range(meta_epochs):
        print("# {}".format(epoch+1))
        train_task_loader.reset()
        # valid_task_loader.reset()
        meta_learner.meta_train(train_task_loader, epoch, writer=writer)
        # meta_learner.meta_test(valid_task_loader, writer=writer)
