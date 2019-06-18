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

    def make_sentence(self, vision, normalize):             # k,100,3,125,125
        inp = torch.cat([vision[:,0], vision[:,-1]], dim=1) # k,6,125,125
        sj = self.emb_net(inp.to(self.device))              # k,20
        if normalize:
            sj = sj.mean(0) # 20,
            sj = sj / torch.norm(sj)
        else:
            sj = sj[0]      # 20,
        return sj

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = torch.dot(q_sj, U_sj)
        fake = torch.dot(q_sj, U_si)
        zero = torch.zeros(1).to(self.device)
        return torch.max(zero, 0.1 - real + fake)

    def meta_train(self, task_loader, num_batch_tasks, epoch, writer=None, train=True):
        device = self.device

        B = num_batch_tasks
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for i, task in enumerate(tqdm(task_loader)): # load tasks one by one

            if i % B == 0:
                if train:
                    self.opt.zero_grad()
                loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0
                U_s_dict, q_s_dict = {}, {}

            U_vision = task["train-vision"][0] # U_n,100,3,125,125
            U_state = task["train-state"][0]   # U_n,100,20
            U_action = task["train-action"][0] # U_n,100,7
            q_vision = task["test-vision"][0]  # q_n,100,3,125,125
            q_state = task["test-state"][0]    # q_n,100,20
            q_action = task["test-action"][0]  # q_n,100,7
            jdx = task["idx"].item()           # 1
            U_n, q_n = len(U_vision), len(q_vision)
            size = U_vision.shape[-1] # 125 or 64

            U_sj = self.make_sentence(U_vision, True)
            q_sj = self.make_sentence(q_vision, False)
            U_s_dict[jdx] = U_sj
            q_s_dict[jdx] = q_sj

            # ---- calc loss_ctr ----
            U_vision = U_vision.view(U_n*100,3,size,size).to(device)
            U_state = U_state.view(U_n*100,20).to(device)
            U_action = U_action.view(U_n*100,7).to(device)
            q_vision = q_vision.view(q_n*100,3,size,size).to(device)
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
            # ----

            if (i+1) % B == 0:

                # ---- calc loss_emb ----
                for (jdx, q_sj), (_, U_sj) in zip(q_s_dict.items(), U_s_dict.items()):
                    for (idx, U_si) in U_s_dict.items():
                        if jdx == idx: continue
                        loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0
                if train:
                    loss_emb.backward()
                loss_emb = loss_emb.item()
                # ----

                if train:
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

    def meta_valid(self, task_loader, num_batch_tasks, epoch, writer):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            self.meta_train(task_loader, num_batch_tasks, epoch, writer, False)

    def make_test_sentence(self, demo_path, emb_net):
        inp = vread(demo_path)
        cv2.imshow("demo", inp[-1][:,:,::-1]); cv2.waitKey(10)
        inp = torch.stack([torch.from_numpy(inp).to("cuda")]) # 1,F,H,W,C
        inp = (inp.permute(0,1,4,2,3).to(torch.float32)-127.5)/127.5 # 1,F,C,H,W
        inp = self.make_sentence(inp, normalize=True) # 20,
        return  inp

    def sim_test(self, env, demo_path):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            return super(TecNets, self).sim_test(env, demo_path, self.make_test_sentence)
