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
    def __init__(self, device, log_dir):
        super(TecNets, self).__init__(device, log_dir)

        # for summury writer, passing by reference
        self.num_iter_tr = [0,]
        self.num_iter_val = [0,]

    def make_emb_dict(self, batch_vision, batch_jdx, normalize):
        device = self.device
        s_dict = {}

        for vision, jdx in zip(batch_vision, batch_jdx):         # k,100,3,125,125
            inps = torch.cat([vision[:,0], vision[:,-1]], dim=1) # k,6,125,125
            sj = self.emb_net(inps.to(device))

            if normalize:
                sj = sj.mean(0)
                sj = sj / torch.norm(sj)
            else:
                sj = sj[0] # for the moment

            s_dict[jdx.item()] = sj

        return s_dict, len(inps)

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = torch.dot(q_sj, U_sj)
        fake = torch.dot(q_sj, U_si)
        zero = torch.zeros(1).to(self.device)
        return torch.max(zero, 0.1 - real + fake)

    def meta_train(self, task_loader, epoch, writer=None, train=True):
        device = self.device
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for batch_task in tqdm(task_loader):
            U_vision = batch_task["train-vision"].to(device) # B,U_n,100,3,125,125
            U_state = batch_task["train-state"].to(device)   # B,U_n,100,20
            U_action = batch_task["train-action"].to(device) # B,U_n,100,7
            q_vision = batch_task["test-vision"].to(device)  # B,q_n,100,3,125,125
            q_state = batch_task["test-state"].to(device)    # B,q_n,100,20
            q_action = batch_task["test-action"].to(device)  # B,q_n,100,7
            batch_jdx = batch_task["idx"].to(device)         # B

            # with torch.no_grad(): # ? # TODO
            U_s, U_n = self.make_emb_dict(U_vision, batch_jdx, True)
            q_s, q_n = self.make_emb_dict(q_vision, batch_jdx, False)
            assert U_s.keys() == q_s.keys(), ""

            # ---- calc loss_emb ----

            loss_emb = 0
            U_sj_list = [] # ctr_net input sentences

            for (jdx, q_sj), (_, U_sj) in zip(q_s.items(), U_s.items()):
                for (idx, U_si) in U_s.items():
                    if jdx == idx: continue
                    loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0

                U_sj_list.append(U_sj)

            # ---- calc loss_ctr ----

            loss_ctr_U, loss_ctr_q = 0, 0

            for i in range(len(batch_task)):
                U_v = U_vision[i].view(U_n*100,3,125,125)
                U_s = U_state[i].view(U_n*100,20)
                U_a = U_action[i].view(U_n*100,7)
                q_v = q_vision[i].view(q_n*100,3,125,125)
                q_s = q_state[i].view(q_n*100,20)
                q_a = q_action[i].view(q_n*100,7)

                U_output = self.ctr_net(U_v, U_sj_list[i].repeat_interleave(100*U_n, dim=0), U_s)
                q_output = self.ctr_net(q_v, U_sj_list[i].repeat_interleave(100*q_n, dim=0), q_s)
                loss_ctr_U += self.loss_fn(U_output, U_a) * len(U_v) * 0.1
                loss_ctr_q += self.loss_fn(q_output, q_a) * len(q_v) * 0.1

            loss = loss_emb + loss_ctr_U + loss_ctr_q

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            loss_emb_list.append(loss_emb.item())
            loss_ctr_U_list.append(loss_ctr_U.item())
            loss_ctr_q_list.append(loss_ctr_q.item())
            loss_list.append(loss.item())

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
        inp = (np.array(vread(demo_path).transpose(0,3,1,2)[[0,-1]].reshape(1,6,125,125)
                        , np.float32)-127.5)/127.5  # 1,6,125,125
        inp = torch.from_numpy(inp).to(self.device)
        inp = emb_net(inp.to(self.device))[0]
        return  inp / torch.norm(inp)
    
    def sim_test(self, env, demo_path):
        return super(TecNets, self).sim_test(env, demo_path, self.make_test_sentence)
