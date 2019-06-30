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
            _temp1 = torch.norm(sj, dim=1, keepdim=True)
            sj = sj / _temp1

            _temp2 = torch.norm(sj, dim=1, keepdim=True)
            print("norm before:", _temp1.mean().data.cpu(), _temp1.std().data.cpu(), \
                  ", norm after:", _temp2.mean().data.cpu(), _temp2.std().data.cpu())

        else:
            assert False, "sentences are always normalized. (by author)"
            sj = sj[:,0]    # N,20
        return sj

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = (q_sj*U_sj).sum(1) # 4032,
        fake = (q_sj*U_si).sum(1) # 4032,
        zero = torch.zeros(1).to(self.device) # 1,
        loss = torch.max(0.1 - real + fake, zero) # 4032,
        return loss

    def meta_train(self, task_loader, epoch, writer=None, train=True):
        device = self.device
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for batch_task in tqdm(task_loader):
            if train:
                self.opt.zero_grad()

            U_visions = batch_task["train-vision"].to(device) # B,U_n,100,3,H,W
            U_states = batch_task["train-state"].to(device)   # B,U_n,100,20
            U_actions = batch_task["train-action"].to(device) # B,U_n,100,7
            q_visions = batch_task["test-vision"].to(device)  # B,q_n,100,3,H,W
            q_states = batch_task["test-state"].to(device)    # B,q_n,100,20
            q_actions = batch_task["test-action"].to(device)  # B,q_n,100,7

            B, U_n, _F, _C, H, W = U_visions.shape
            q_n = q_visions.shape[1]

            U_s = self.make_sentences(U_visions, True) # B,20
            q_s = self.make_sentences(q_visions, True) # B,20

            loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0

            q_sj_list, U_sj_list, U_si_list = [], [], []

            # ---- calc loss_emb ----

            for jdx in range(B):
                for idx in range(B):
                    if jdx == idx: continue
                    q_sj_list.append(q_s[jdx])
                    U_sj_list.append(U_s[jdx])
                    U_si_list.append(U_s[idx])

            q_sj = torch.stack(q_sj_list) # B*(B-1), 20
            U_sj = torch.stack(U_sj_list) # B*(B-1), 20
            U_si = torch.stack(U_si_list) # B*(B-1), 20

            loss_emb += torch.mean(self.cos_hinge_loss(q_sj, U_sj, U_si)) * 1.0 / (B*100.)

            # ---- calc loss_ctr ----

            U_vision = U_visions.view(B*U_n*100,3,H,W)
            U_state = U_states.view(B*U_n*100,20)
            U_action = U_actions.view(B*U_n*100,7)
            q_vision = q_visions.view(B*q_n*100,3,H,W)
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
