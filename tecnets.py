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
    def __init__(self, device, log_dir=None, lr=None):
        super(TecNets, self).__init__(device, log_dir, lr)

    def make_sentence(self, vision, normalize):
        N, k, _F, _C, H, W = vision.shape                       # N,k,100,3,H,W
        inp = torch.cat([vision[:,:,0], vision[:,:,-1]], dim=2) # N,k,6,H,W
        sj = self.emb_net(inp.view(N*k,6,H,W)).view(N,k,20)     # N,k,20
        if normalize:
            sj = sj.mean(1) # N,20
            sj = sj / torch.norm(sj, 1)
        else:
            sj = sj[:,0]    # N,20
        return sj

    def cos_hinge_loss(self, q_sj_list, U_sj_list, U_si_list):
        q_sj = torch.stack(q_sj_list) # 4032, 20
        U_sj = torch.stack(U_sj_list) # 4032, 20
        U_si = torch.stack(U_si_list) # 4032, 20

        real = (q_sj*U_sj).sum(1) # 4032,
        fake = (q_sj*U_si).sum(1) # 4032,
        zero = torch.zeros(1).to(self.device) # 1,

        loss = torch.max(0.1 - real + fake, zero) # 4032,
        return loss

    def meta_train(self, task_loader, num_batch_tasks, num_load_tasks, epoch, writer=None, train=True):
        device = self.device

        B = num_batch_tasks
        N = num_load_tasks  # e.g. 16
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        # N tasks * (n_tasks/N)iter # e.g. 16task * 44iter
        for i, tasks in enumerate(tqdm(task_loader)):

            if (i*N) % B == 0:
                if train:
                    self.opt.zero_grad()
                loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0
                U_s_list, q_s_list = [], []

            U_vision = tasks["train-vision"] # N,U_n,100,3,125,125
            U_state = tasks["train-state"]   # N,U_n,100,20
            U_action = tasks["train-action"] # N,U_n,100,7
            q_vision = tasks["test-vision"]  # N,q_n,100,3,125,125
            q_state = tasks["test-state"]    # N,q_n,100,20
            q_action = tasks["test-action"]  # N,q_n,100,7
            U_n, q_n = U_vision.shape[1], q_vision.shape[1]
            size = U_vision.shape[5] # 125 or 64

            U_sj = self.make_sentence(U_vision, normalize=True)  # N,20
            q_sj = self.make_sentence(q_vision, normalize=False) # N,20
            U_s_list.append(U_sj)
            q_s_list.append(q_sj)

            # ---- calc loss_ctr ----
            U_vision = U_vision.view(N*U_n*100,3,size,size).to(device) # N*U_n*100,C,H,W
            U_state = U_state.view(N*U_n*100,20).to(device)            # N*U_n*100,20
            U_action = U_action.view(N*U_n*100,7).to(device)           # N*U_n*100,7
            q_vision = q_vision.view(N*q_n*100,3,size,size).to(device) # N*q_n*100,C,H,W
            q_state = q_state.view(N*q_n*100,20).to(device)            # N*q_n*100,20
            q_action = q_action.view(N*q_n*100,7).to(device)           # N*q_n*100,7
            U_sj_U_inp = U_sj.repeat_interleave(100*U_n, dim=0)        # N*U_n*100,20
            U_sj_q_inp = U_sj.repeat_interleave(100*q_n, dim=0)        # N*q_n*100,20

            U_out = self.ctr_net(U_vision, U_sj_U_inp, U_state)
            _loss_ctr_U = self.loss_fn(U_out, U_action) * len(U_vision) * 0.1 / (B*100)
            if train:
                _loss_ctr_U.backward(retain_graph=True) # memory saving
            loss_ctr_U += _loss_ctr_U.item()

            q_out = self.ctr_net(q_vision, U_sj_q_inp, q_state)
            _loss_ctr_q = self.loss_fn(q_out, q_action) * len(q_vision) * 0.1 / (B*100)
            if train:
                _loss_ctr_q.backward(retain_graph=True) # memory saving
            loss_ctr_q += _loss_ctr_q.item()
            # ----

            if ((i+1)*N) % B == 0:

                # don't convert into list. graph informations will be lost. (and an error will occur)
                U_s_list = torch.cat(U_s_list, 0) # N*(B/N),20 -> N,20
                q_s_list = torch.cat(q_s_list, 0) # N*(B/N),20 -> N,20

                q_sj_list, U_sj_list, U_si_list = [], [], []

                # ---- calc loss_emb ----
                """
                for jdx, (q_sj, U_sj) in enumerate(zip(q_s_list, U_s_list)):
                    for idx, U_si in enumerate(U_s_list):
                        if jdx == idx: continue
                        q_sj_list.append(q_sj)
                        U_sj_list.append(U_sj)
                        U_si_list.append(U_si)
                _loss_emb = torch.sum(self.cos_hinge_loss(q_sj_list, U_sj_list, U_si_list)) * 1.0

                if train:
                    _loss_emb.backward()
                """
                loss_emb = 0 # _loss_emb.item()
                # ----

                if train:
                    torch.nn.utils.clip_grad_value_(self.emb_net.parameters(), 10)
                    torch.nn.utils.clip_grad_value_(self.ctr_net.parameters(), 10)
                    self.opt.step()

                loss = loss_emb + loss_ctr_U + loss_ctr_q
                loss_emb_list.append(loss_emb)
                loss_ctr_U_list.append(loss_ctr_U)
                loss_ctr_q_list.append(loss_ctr_q)
                loss_list.append(loss)

        # -- end all tasks

        loss_emb = np.mean(loss_emb_list)
        loss_ctr_U = np.mean(loss_ctr_U_list)
        loss_ctr_q = np.mean(loss_ctr_q_list)
        loss = np.mean(loss_list)

        print("loss:", loss, ", loss_ctr_U:", loss_ctr_U, ", loss_ctr_q:", loss_ctr_q, ", loss_emb:", loss_emb)

        if writer:
            writer.add_scalar('loss_emb', loss_emb, epoch)
            writer.add_scalar('loss_ctr_U', loss_ctr_U, epoch)
            writer.add_scalar('loss_ctr_q', loss_ctr_q, epoch)
            writer.add_scalar('loss_all', loss, epoch)

        if train:
            path = self.log_dir.split("/")[-1]+"_epoch"+str(epoch)+"_"+f'{loss:.4f}'
            self.save_emb_net(self.log_dir+"/"+path+"_emb.pt")
            self.save_ctr_net(self.log_dir+"/"+path+"_ctr.pt")
            self.save_opt(self.log_dir+"/"+path+"_opt.pt")

    def meta_valid(self, task_loader, num_batch_tasks, num_load_tasks, epoch, writer):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            self.meta_train(task_loader, num_batch_tasks, num_load_tasks, epoch, writer, False)

    # TODO: n-shot
    def make_test_sentence(self, demo_path, emb_net):
        inp = vread(demo_path)
        cv2.imshow("demo", inp[-1][:,:,::-1]); cv2.waitKey(10)
        inp = torch.stack([torch.from_numpy(inp).to("cuda")]) # 1,F,H,W,C
        inp = inp.permute(0,1,4,2,3).to(torch.float32) / 255.0 # 1,F,C,H,W
        inp = torch.stack([inp]) # 1,1,F,C,H,W # N,k,F,C,H,W
        inp = self.make_sentence(inp, normalize=True)[0] # 20,
        return  inp

    def sim_test(self, env, demo_path):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            return super(TecNets, self).sim_test(env, demo_path, self.make_test_sentence)
