import torch
import cv2
import itertools
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from meta_learner import MetaLearner
from utils import vread

import time

class TecNets(MetaLearner):
    def __init__(self, device, log_dir=None):
        super(TecNets, self).__init__(device, log_dir)

        # for summury writer, passing by reference
        self.num_iter_tr = [0,]
        self.num_iter_val = [0,]

    def make_sentences(self, vision, normalize):
        N, k, _F, _C, H, W = vision.shape        # N,k,100,3,H,W
        inp = vision[:,:,[0,-1]].view(N*k,6,H,W) # N,k,6,H,W
        sj = self.emb_net(inp).view(N,k,20)      # N,k,20
        if normalize:
            sj = sj.mean(1) # N,20
            sj = sj / torch.norm(sj, 1)
        else:
            sj = sj[:,0]    # N,20
        return sj

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = torch.dot(q_sj, U_sj)
        fake = torch.dot(q_sj, U_si)
        zero = torch.zeros(1).to(self.device)
        return torch.max(zero, 0.1 - real + fake)

    def meta_train(self, task_loader, epoch, writer=None, train=True):
        device = self.device
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for batch_task in tqdm(task_loader):
            if train:
                self.opt.zero_grad()

            U_visions = batch_task["train-vision"].to(device) # B,U_n,100,3,125,125
            U_states = batch_task["train-state"].to(device)   # B,U_n,100,20
            U_actions = batch_task["train-action"].to(device) # B,U_n,100,7
            q_visions = batch_task["test-vision"].to(device)  # B,q_n,100,3,125,125
            q_states = batch_task["test-state"].to(device)    # B,q_n,100,20
            q_actions = batch_task["test-action"].to(device)  # B,q_n,100,7
            jdxs = batch_task["idx"].to(device)               # B

            B = len(U_visions)
            U_n, q_n = U_visions.shape[1], q_visions.shape[1]

            U_s = self.make_sentences(U_visions, True) # B,20
            # q_s = self.make_sentences(q_visions, True) # B,20

            loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0

            # lambda_emb = 0 ver. (original paper reported 58.56% success)
            # ---- calc loss_emb ----

            # for (jdx, q_sj), (_, U_sj) in zip(q_s.items(), U_s.items()): # for idx in range(64): ??
            #     for (idx, U_si) in U_s.items():
            #         if jdx == idx: continue
            #         loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0 / (4032*B*100)

            # ---- calc loss_ctr ----

            U_vision = U_visions.view(B*U_n*100,3,125,125)
            U_state = U_states.view(B*U_n*100,20)
            U_action = U_actions.view(B*U_n*100,7)
            q_vision = q_visions.view(B*q_n*100,3,125,125)
            q_state = q_states.view(B*q_n*100,20)
            q_action = q_actions.view(B*q_n*100,7)
            U_sj_U_inp = U_s.repeat_interleave(100*U_n, dim=0) # N*100*U_n, 20
            U_sj_q_inp = U_s.repeat_interleave(100*q_n, dim=0)

            U_out = self.ctr_net(U_vision, U_sj_U_inp, U_state)
            loss_ctr_U += self.loss_fn(U_out, U_action) * len(U_vision) * 0.1 / (B*100.)

            q_out = self.ctr_net(q_vision, U_sj_q_inp, q_state)
            loss_ctr_q += self.loss_fn(q_out, q_action) * len(q_vision) * 0.1 / (B*100.)

            loss = loss_emb + loss_ctr_U + loss_ctr_q

            if train:
                # self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            loss_emb_list.append(loss_emb)
            loss_ctr_U_list.append(loss_ctr_U.item())
            loss_ctr_q_list.append(loss_ctr_q.item())
            loss_list.append(loss.item())

            import subprocess
            _cmd = "nvidia-smi"
            subprocess.call(_cmd.split())

        # -- end batch tasks

        loss_emb = np.mean(loss_emb_list)
        loss_ctr_U = np.mean(loss_ctr_U_list)
        loss_ctr_q = np.mean(loss_ctr_q_list)
        loss = np.mean(loss_list)

        num_iter = self.num_iter_tr if train else self.num_iter_val
        if writer:
            writer.add_scalar('loss_emb', loss_emb, num_iter[0])
            writer.add_scalar('loss_ctr_U', loss_ctr_U, num_iter[0])
            writer.add_scalar('loss_ctr_q', loss_ctr_q, num_iter[0])
            writer.add_scalar('loss_all', loss, num_iter[0])
        num_iter[0]+=1

        if train:
            self.save_emb_net(self.log_dir+"/emb_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")
            self.save_ctr_net(self.log_dir+"/ctr_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")
            self.save_opt(self.log_dir+"/opt_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")

    def meta_test(self, task_loader, writer=None):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            self.meta_train(task_loader=task_loader, epoch=None, writer=writer, train=False)

    def make_test_sentence(self, demo_path, emb_net):
        inp = vread(demo_path)
        # cv2.imshow("demo", inp[-1][:,:,::-1]); cv2.waitKey(10)
        inp = torch.stack([torch.from_numpy(inp).to("cuda")]) # 1,F,H,W,C
        inp = (inp.permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5 # 1,F,C,H,W
        inp = torch.stack([inp]) # 1,1,F,C,H,W # N,k,F,C,H,W
        inp = self.make_sentences(inp, normalize=True) # 1,20,
        return  inp

    def sim_test(self, env, demo_path):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            return super(TecNets, self).sim_test(env, demo_path, self.make_test_sentence)
