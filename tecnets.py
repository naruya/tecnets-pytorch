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

        # for summury writer, for passing by reference
        self.num_iter_tr = [0,]
        self.num_iter_val = [0,]

    def make_emb_dict(self, batch_task):
        device = self.device
        U_s, q_s = {}, {} # support/query_sentence dict

        for task in batch_task:
            U_inps = task['train']['vision']                       # U_n,100,3,125,125
            q_inps = task['test']['vision']                        # q_n,100,3,125,125
            U_inps = torch.cat([U_inps[:,0], U_inps[:,-1]], dim=1) # U_n,6,125,125
            q_inps = torch.cat([q_inps[:,0], q_inps[:,-1]], dim=1) # q_n,6,125,125

            U_sj = self.emb_net(U_inps.to(device)).mean(0)
            U_sj = U_sj / torch.norm(U_sj)
            q_sj = self.emb_net(q_inps.to(device))[0]

            jdx = task['task_idx']
            U_s[jdx] = U_sj
            q_s[jdx] = q_sj

        return U_s, q_s, len(U_inps), len(q_inps)

    def cos_hinge_loss(self, q_sj, U_sj, U_si):
        real = torch.dot(q_sj, U_sj)
        fake = torch.dot(q_sj, U_si)
        zero = torch.zeros(1).to(self.device)
        loss = torch.max(zero, 0.1 - real + fake)
        return loss

    def meta_train(self, task_loader, epoch, writer=None, train=True):
        device = self.device
        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        start = time.time() # 1

        for batch_task in tqdm(task_loader):

            elapsed_time = time.time() - start
            print ("1. elapsed_time:{0}".format(elapsed_time) + "[sec]")

            batch_task_pre, batch_task_emb, batch_task_ctr = itertools.tee(batch_task, 3)

            U_s, q_s, U_n, q_n = self.make_emb_dict(batch_task_pre)

            # ---- calc loss_emb ----

            loss_emb = 0
            U_sj_list = [] # ctr_net input sentences

            start = time.time() # 2

            for task in batch_task_emb:
                jdx = task['task_idx']
                U_sj = U_s[jdx]
                q_sj = q_s[jdx]

                for idx, U_si in list(U_s.items()):
                    if jdx == idx: continue
                    loss_emb += self.cos_hinge_loss(q_sj, U_sj, U_si) * 1.0

                U_sj_list.append(U_sj)

            elapsed_time = time.time() - start
            print ("2. elapsed_time:{0}".format(elapsed_time) + "[sec]")

            # ---- calc loss_ctr ----

            start = time.time() # 3

            U_vision, U_state, U_action, \
                q_vision, q_state, q_action = self.stack_demos(batch_task_ctr)
            U_sj = torch.stack(U_sj_list)

            elapsed_time = time.time() - start
            print ("3. elapsed_time:{0}".format(elapsed_time) + "[sec]")

            U_output = self.ctr_net(U_vision, U_sj.repeat_interleave(100*U_n, dim=0), U_state)
            q_output = self.ctr_net(q_vision, U_sj.repeat_interleave(100*q_n, dim=0), q_state)
            loss_ctr_U = self.loss_fn(U_output, U_action) * len(U_vision) * 0.1
            loss_ctr_q = self.loss_fn(q_output, q_action) * len(q_vision) * 0.1

            loss = loss_emb + loss_ctr_U + loss_ctr_q

            start = time.time() # 4

            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            elapsed_time = time.time() - start
            print ("4. elapsed_time:{0}".format(elapsed_time) + "[sec]")

            loss_emb_list.append(loss_emb.item())
            loss_ctr_U_list.append(loss_ctr_U.item())
            loss_ctr_q_list.append(loss_ctr_q.item())
            loss_list.append(loss.item())

            start = time.time() # 1

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
