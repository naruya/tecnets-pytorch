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

    def make_sentence(self, vision, normalize):
        B, k, _F, _C, H, W = vision.shape                       # B,k,100,3,125,125
        inp = torch.cat([vision[:,:,0], vision[:,:,-1]], dim=2) # B,k,6,125,125
        sj = self.emb_net(inp.view(B*k,6,H,W)).view(B,k,20)     # B,k,20
        if normalize:
            sj = sj.mean(1) # B,20
            sj = sj / torch.norm(sj, 1)
        else:
            sj = sj[:,0]    # B,20
        return sj

    def cos_hinge_loss(self, q_sj_list, U_sj_list, U_si_list):
        q_sj = torch.stack(q_sj_list) # 4032, 20
        U_sj = torch.stack(U_sj_list) # 4032, 20
        U_si = torch.stack(U_si_list) # 4032, 20

        real = (q_sj*U_sj).sum(1) # 4032,
        fake = (q_sj*U_si).sum(1) # 4032,
        zero = torch.zeros(1).to(self.device) # 1,

        loss = torch.max(0.1 - real + fake, zero)
        return loss # 4032,

    def meta_train(self, task_loader, num_batch_tasks, num_load_tasks, epoch, writer=None, train=True):
        device = self.device

        B = num_load_tasks
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for i, tasks in enumerate(tqdm(task_loader)): # 32

            if (i*B) % num_batch_tasks == 0:
                if train:
                    self.opt.zero_grad()
                loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0
                U_s_list, q_s_list = [], []

            U_vision = tasks["train-vision"] # B,U_n,100,C,H,W
            U_state = tasks["train-state"]   # B,U_n,100,20
            U_action = tasks["train-action"] # B,U_n,100,7
            q_vision = tasks["test-vision"]  # B,q_n,100,C,H,W
            q_state = tasks["test-state"]    # B,q_n,100,20
            q_action = tasks["test-action"]  # B,q_n,100,7
            U_n, q_n = U_vision.shape[1], q_vision.shape[1]
            size = U_vision.shape[5] # 125 or 64

            U_sj = self.make_sentence(U_vision, normalize=True)  # B,20
            q_sj = self.make_sentence(q_vision, normalize=False) # B,20
            U_s_list.append(U_sj)
            q_s_list.append(q_sj)

            # ---- calc loss_ctr ----
            U_vision = U_vision.view(B*U_n*100,3,size,size).to(device) # B*U_n*100,C,H,W
            U_state = U_state.view(B*U_n*100,20).to(device)            # B*U_n*100,20
            U_action = U_action.view(B*U_n*100,7).to(device)           # B*U_n*100,7
            q_vision = q_vision.view(B*q_n*100,3,size,size).to(device) # B*q_n*100,C,H,W
            q_state = q_state.view(B*q_n*100,20).to(device)            # B*q_n*100,20
            q_action = q_action.view(B*q_n*100,7).to(device)           # B*q_n*100,7
            U_sj_U_inp = U_sj.repeat_interleave(U_n*100, dim=0)        # B*U_n*100,20
            U_sj_q_inp = U_sj.repeat_interleave(q_n*100, dim=0)        # B*q_n*100,20

            U_out = self.ctr_net(U_vision, U_sj_U_inp, U_state)
            _loss_ctr_U = self.loss_fn(U_out, U_action) * len(U_vision) * 0.1
            loss_ctr_U += _loss_ctr_U

            q_out = self.ctr_net(q_vision, U_sj_q_inp, q_state)
            _loss_ctr_q = self.loss_fn(q_out, q_action) * len(q_vision) * 0.1
            loss_ctr_q += _loss_ctr_q
            # ----

            if ((i+1)*B) % num_batch_tasks == 0:

                U_s_list = torch.cat(U_s_list, 0)
                q_s_list = torch.cat(q_s_list, 0)

                q_sj_list, U_sj_list, U_si_list = [], [], []

                # ---- calc loss_emb ----
                for jdx, (q_sj, U_sj) in enumerate(zip(q_s_list, U_s_list)):
                    for idx, U_si in enumerate(U_s_list):
                        if jdx == idx: continue
                        q_sj_list.append(q_sj)
                        U_sj_list.append(U_sj)
                        U_si_list.append(U_si)
                loss_emb = torch.sum(self.cos_hinge_loss(q_sj_list, U_sj_list, U_si_list)) * 1.0

                loss = loss_emb + loss_ctr_U + loss_ctr_q
                if train:
                    loss.backward()

                loss = loss.item()
                loss_emb = loss_emb.item()
                loss_ctr_q = loss_ctr_q.item()
                loss_ctr_U = loss_ctr_U.item()
                # ----

                if train:
                    self.opt.step()

                # loss = loss_emb + loss_ctr_U + loss_ctr_q
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

    def meta_test(self, task_loader, num_batch_tasks, num_load_tasks, epoch, writer):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            self.meta_train(task_loader, num_batch_tasks, num_load_tasks, epoch, writer, False)

    def make_test_sentence(self, demo_path, emb_net):
        inp = (vread(demo_path).transpose(0,3,1,2)[[0,-1]].reshape(1,6,125,125).astype(np.float32)
                        -127.5)/127.5  # 1,6,125,125
        inp = torch.from_numpy(inp).to(self.device)
        inp = emb_net(inp.to(self.device))[0]
        return  inp / torch.norm(inp)
    
    def sim_test(self, env, demo_path):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            return super(TecNets, self).sim_test(env, demo_path, self.make_test_sentence)
