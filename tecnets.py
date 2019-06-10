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

    def make_sentence(self, vision, normalize):             # k,100,3,125,125
        inp = torch.cat([vision[:,0], vision[:,-1]], dim=1) # k,6,125,125
        sj = self.emb_net(inp.to(self.device))
        if normalize:
            sj = sj.mean(0)
            sj = sj / torch.norm(sj)
        else:
            sj = sj[0]
        return sj

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = torch.dot(q_sj, U_sj)
        fake = torch.dot(q_sj, U_si)
        zero = torch.zeros(1).to(self.device)
        return torch.max(zero, 0.1 - real + fake)

    def meta_train(self, task_loader, num_batch_tasks, epoch, writer=None, train=True):
        device = self.device
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for _i in range(len(task_loader.dataset) // num_batch_tasks):
            if train:
                self.opt.zero_grad()
            loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0
            U_s_dict, q_s_dict = {}, {}

            for _j in range(num_batch_tasks):
                print("_i, _j", _i, _j)
                task = iter(task_loader).next() # load tasks one by one

                U_vision = task["train-vision"][0] # U_n,100,3,125,125
                U_state = task["train-state"][0]   # U_n,100,20
                U_action = task["train-action"][0] # U_n,100,7
                q_vision = task["test-vision"][0]  # q_n,100,3,125,125
                q_state = task["test-state"][0]    # q_n,100,20
                q_action = task["test-action"][0]  # q_n,100,7
                jdx = task["idx"].item()           # 1

                U_n, q_n = len(U_vision), len(q_vision)
                print("U_n, q_n", U_n, q_n)

                U_sj = self.make_sentence(U_vision, True)
                q_sj = self.make_sentence(q_vision, False)
                U_s_dict[jdx] = U_sj
                q_s_dict[jdx] = q_sj

                # ---- calc loss_ctr ----

                U_vision = U_vision.view(U_n*100,3,125,125).to(device)
                U_state = U_state.view(U_n*100,20).to(device)
                U_action = U_action.view(U_n*100,7).to(device)
                q_vision = q_vision.view(q_n*100,3,125,125).to(device)
                q_state = q_state.view(q_n*100,20).to(device)
                q_action = q_action.view(q_n*100,7).to(device)
                U_sj_U_inp = U_sj.repeat_interleave(100*U_n, dim=0)
                U_sj_q_inp = U_sj.repeat_interleave(100*q_n, dim=0)

                U_out = self.ctr_net(U_vision, U_sj_U_inp, U_state)
                _loss_ctr_U = self.loss_fn(U_out, U_action) * len(U_vision) * 0.1
                if train:
                    _loss_ctr_U.backward(retain_graph=True) # memory saving
                loss_ctr_U += _loss_ctr_U.item()

                q_out = self.ctr_net(q_vision, U_sj_q_inp, q_state)
                _loss_ctr_q = self.loss_fn(q_out, q_action) * len(q_vision) * 0.1
                if train:
                    _loss_ctr_q.backward(retain_graph=True) # memory saving
                loss_ctr_q += _loss_ctr_q.item()

            # ---- calc loss_emb ----

            for (jdx, q_sj), (_, U_sj) in zip(q_s_dict.items(), U_s_dict.items()):
                for (idx, U_si) in U_s_dict.items():
                    if jdx == idx: continue
                    loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0

            if train:
                loss_emb.backward()
            loss_emb = loss_emb.item()

            loss = loss_emb + loss_ctr_U + loss_ctr_q

            if train:
                self.opt.step()

            loss_emb_list.append(loss_emb)
            loss_ctr_U_list.append(loss_ctr_U)
            loss_ctr_q_list.append(loss_ctr_q)
            loss_list.append(loss)

        # -- end all tasks

        loss_emb = np.mean(loss_emb_list)
        loss_ctr_U = np.mean(loss_ctr_U_list)
        loss_ctr_q = np.mean(loss_ctr_q_list)
        loss = np.mean(loss_list)

        if writer:
            writer.add_scalar('loss_emb', loss_emb, epoch)
            writer.add_scalar('loss_ctr_U', loss_ctr_U, epoch)
            writer.add_scalar('loss_ctr_q', loss_ctr_q, epoch)
            writer.add_scalar('loss_all', loss, epoch)

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
