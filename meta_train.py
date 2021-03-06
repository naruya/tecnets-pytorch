# export CUDA_VISIBLE_DEVICES="2, 3"

import argparse
import os
import random
import subprocess
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

from taskset import MILTaskset
from torch.utils.data import DataLoader as TaskLoader
from tecnets import TecNets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--demo_dir', type=str, default='sim_push/')
    parser.add_argument('--state_path', type=str, default="scale_and_bias_sim_push.pkl")
    parser.add_argument('--num_batch_tasks', type=int, default=64)
    parser.add_argument('--num_load_tasks', type=int, default=4)
    parser.add_argument('--train_n_shot', type=int, default=1)
    parser.add_argument('--test_n_shot', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
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

    _cmd = "git rev-parse --short HEAD"
    commit = subprocess.check_output(_cmd.split()).strip().decode('utf-8')

    _cmd = "git rev-parse --abbrev-ref HEAD"
    branch = subprocess.check_output(_cmd.split()).strip().decode('utf-8')
    branch = "-".join(branch.split("/"))

    log_dir = "./logs/" + datetime.now().strftime('%m%d-%H%M%S') \
                        + "_" + branch \
                        + "_" + commit \
                        + "_B"+str(args.num_batch_tasks) \
                        + "_size125" \
                        + "_shot" + str(args.train_n_shot) + "-" + str(args.test_n_shot) \
                        + "_lr" + str(args.lr)
    print(log_dir)
    os.mkdir(log_dir)

    train_writer = SummaryWriter("./runs/"+log_dir.split("/")[-1]+"_train")
    valid_writer = SummaryWriter("./runs/"+log_dir.split("/")[-1]+"_valid")

    meta_learner = TecNets(device=device, log_dir=log_dir, lr=args.lr)

    if args.resume_epoch:
        print("resuming...")
        print(args.emb_path)
        print(args.ctr_path)
        print(args.opt_path)
        meta_learner.resume(args.emb_path, args.ctr_path, args.opt_path, device)
        resume_epoch = args.resume_epoch
    else:
        resume_epoch = 0

    # for memory saving, for convenience, task_batch_size=1 (See tecnets*.py meta_train)
    train_task_loader = TaskLoader(MILTaskset(
        demo_dir=args.demo_dir, state_path=args.state_path, num_batch_tasks=args.num_batch_tasks,
        train_n_shot=args.train_n_shot, test_n_shot=1, val_size=args.num_batch_tasks),
                                   batch_size=args.num_load_tasks, shuffle=True,
                                   num_workers=4, pin_memory=True)
    valid_task_loader = TaskLoader(MILTaskset(
        demo_dir=args.demo_dir, state_path=args.state_path, num_batch_tasks=args.num_batch_tasks,
        train_n_shot=args.train_n_shot, test_n_shot=1, valid=True, val_size=args.num_batch_tasks),
                                   batch_size=args.num_load_tasks, shuffle=True,
                                   num_workers=4, pin_memory=True)

    meta_epochs = 400000 # 400000/11

    for epoch in range(resume_epoch, meta_epochs):
        print("# {}".format(epoch+1))
        meta_learner.meta_train(train_task_loader, args.num_batch_tasks, args.num_load_tasks, epoch, writer=train_writer)
        meta_learner.meta_valid(valid_task_loader, args.num_batch_tasks, args.num_load_tasks, epoch, writer=valid_writer)
