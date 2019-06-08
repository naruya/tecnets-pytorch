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

    def make_emb_dict(self, visions, jdxs, normalize):
        device = self.device
        s_dict = {}

        for vision, jdx in zip(visions, jdxs):                   # k,100,3,125,125
            inp = torch.cat([vision[:,0], vision[:,-1]], dim=1) # k,6,125,125
            sj = self.emb_net(inp.to(device))

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
            U_visions = batch_task["train-vision"] # B,U_n,100,3,125,125
            U_states = batch_task["train-state"]   # B,U_n,100,20
            U_actions = batch_task["train-action"] # B,U_n,100,7
            q_visions = batch_task["test-vision"]  # B,q_n,100,3,125,125
            q_states = batch_task["test-state"]    # B,q_n,100,20
            q_actions = batch_task["test-action"]  # B,q_n,100,7
            jdxs = batch_task["idx"]               # B

            # with torch.no_grad(): # ? # TODO
            U_s, U_n = self.make_emb_dict(U_visions, jdxs, True)
            q_s, q_n = self.make_emb_dict(q_visions, jdxs, False)
            assert U_s.keys() == q_s.keys(), ""

            loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0

            # ---- calc loss_emb ----

            U_sj_list = [] # ctr_net input sentences

            for (jdx, q_sj), (_, U_sj) in zip(q_s.items(), U_s.items()):
                for (idx, U_si) in U_s.items():
                    if jdx == idx: continue
                    loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0

                U_sj_list.append(U_sj) # prepare for ctr_net forwarding

            # ---- calc loss_ctr ----

            for i in range(len(batch_task)):
                U_vision = U_visions[i].view(U_n*100,3,125,125).to(device)
                U_state = U_states[i].view(U_n*100,20).to(device)
                U_action = U_actions[i].view(U_n*100,7).to(device)
                q_vision = q_visions[i].view(q_n*100,3,125,125).to(device)
                q_state = q_states[i].view(q_n*100,20).to(device)
                q_action = q_actions[i].view(q_n*100,7).to(device)
                U_sj_U_inp = U_sj_list[i].repeat_interleave(100*U_n, dim=0)
                U_sj_q_inp = U_sj_list[i].repeat_interleave(100*q_n, dim=0)

                U_out = self.ctr_net(U_vision, U_sj_U_inp, U_state)
                q_out = self.ctr_net(q_vision, U_sj_q_inp, q_state)
                loss_ctr_U += self.loss_fn(U_out, U_action) * len(U_v) * 0.1
                loss_ctr_q += self.loss_fn(q_out, q_action) * len(q_v) * 0.1

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
