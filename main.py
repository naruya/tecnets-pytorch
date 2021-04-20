import argparse
import os
import random

import torch


from dataset import TecnetsDataset
from torch.utils.data import DataLoader
from tecnets import TecNets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--demo_dir', type=str, default='/root/datasets/mil_sim_push/')
    parser.add_argument('--num_batch_tasks', type=int, default=32)
    parser.add_argument('--num_load_tasks', type=int, default=4)
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
    train_task_loader = DataLoader(TecnetsDataset(demo_dir=args.demo_dir),
                                   batch_size=args.num_load_tasks,
                                   shuffle=True,
                                   num_workers=os.cpu_count(),
                                   pin_memory=True,
                                   drop_last=True)
    valid_task_loader = DataLoader(TecnetsDataset(demo_dir=args.demo_dir),
                                   batch_size=args.num_load_tasks,
                                   shuffle=True,
                                   num_workers=os.cpu_count(),
                                   pin_memory=True,
                                   drop_last=True)
    
    torch.backends.cudnn.benchmark = True

    meta_epochs = 2  # 400000/11

    for epoch in range(resume_epoch, meta_epochs):
        print("# {}".format(epoch + 1))
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        meta_learner.meta_train(task_loader=train_task_loader, num_batch_tasks=args.num_batch_tasks, num_load_tasks=args.num_load_tasks, epoch=epoch)
        # print(prof.table())
        # prof.export_chrome_trace('./logs/profile.json')
        meta_learner.meta_valid(task_loader=valid_task_loader, num_batch_tasks=args.num_batch_tasks, num_load_tasks=args.num_load_tasks, epoch=epoch)
