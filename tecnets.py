import torch
import cv2
import itertools
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from meta_learner import MetaLearner
from utils import vread

class TecNets(MetaLearner):
    def __init__(self, device, state_path, demo_dir):
        super(TecNets, self).__init__(device, state_path, demo_dir)

    def meta_train(self, task_loader, log_dir, epoch, writer=None, ctr_scale=None):
        device = self.device

        for batch_task in tqdm(task_loader):

            batch_task_pre_emb, batch_task_post_emb, \
                batch_task_pre_ctr, batch_task_post_ctr = itertools.tee(batch_task, 4)

            # ----------------------------------------------------------------
            # make sentences
            # ----------------------------------------------------------------
            
            U_s = {} # support_sentence
            q_s = {} # query_sentence

            for task in batch_task_pre_emb:

                U_inps = task['train']['vision'] # 64, 100*n

                # n,100,3,125,125
                U_inps = torch.cat([U_inps[:,0], U_inps[:,-1]], dim=1)
                # _U_inps = np.array(U_inps.numpy()[0] * 127.5+127.5, np.uint8)
                # _U_inps = _U_inps[:3].transpose(1,2,0)
                # cv2.imshow("inp[0]", _U_inps[:,:,::-1]); cv2.waitKey(1)
                # _U_inps = np.array(U_inps.numpy()[0] * 127.5+127.5, np.uint8)
                # _U_inps = _U_inps[3:].transpose(1,2,0)
                # cv2.imshow("inp[-1]", _U_inps[:,:,::-1]); Scv2.waitKey(1)

                q_inps = task['test']['vision']
                q_inps = torch.cat([q_inps[:,0], q_inps[:,-1]], dim=1)

                U_sj = self.emb_net(U_inps.to(device)).mean(0)
                U_sj = U_sj / torch.norm(U_sj)
                q_sj = self.emb_net(q_inps.to(device))[0]

                jdx = task['task_idx']
                U_s[jdx] = U_sj
                q_s[jdx] = q_sj

            self.opt.zero_grad()

            # ----------------------------------------------------------------
            # calc loss_emb
            # ----------------------------------------------------------------

            loss_emb = 0

            for task in batch_task_post_emb:

                jdx = task['task_idx']
                U_sj = U_s[jdx]
                q_sj = q_s[jdx]

                for idx, U_si in list(U_s.items()):
                    if jdx == idx: continue

                    real = torch.dot(q_sj, U_sj)
                    fake = torch.dot(q_sj, U_si)
                    zero = torch.zeros(1).to(device)
                    loss_emb += torch.max(zero, 0.1 - real + fake)*1.0 # TODO:λ
            
            loss_emb.backward(retain_graph=True) # memory saving

            # ----------------------------------------------------------------
            # prepare inputs for loss_ctr
            # ----------------------------------------------------------------

            U_sj_list = []
            q_sj_list = []

            for task in batch_task_pre_ctr:

                jdx = task['task_idx']
                U_sj_list.append(U_s[jdx])
                q_sj_list.append(q_s[jdx])

            # 64x20 -> 6400x20
            U_sj = torch.stack(U_sj_list).to(device).repeat_interleave(100, dim=0)
            q_sj = torch.stack(q_sj_list).to(device).repeat_interleave(100, dim=0)

            # ----------------------------------------------------------------
            # calc loss_ctr
            # ----------------------------------------------------------------

            U_vision, U_state, U_action, \
                q_vision, q_state, q_action = self.stack_demos(batch_task_post_ctr)

            U_state = torch.matmul(U_state, self.scale) + self.bias
            q_state = torch.matmul(q_state, self.scale) + self.bias

            output = self.ctr_net(U_vision, U_sj, U_state)
            loss_ctr_U = self.loss_fn(output, U_action)*len(U_vision)*0.1
            loss_ctr_U.backward(retain_graph=True) # memory saving

            output = self.ctr_net(q_vision, U_sj, q_state)
            loss_ctr_q = self.loss_fn(output, q_action)*len(q_vision)*0.1
            loss_ctr_q.backward() # memory saving

            loss = loss_emb.item() + loss_ctr_U.item() + loss_ctr_q.item()

            self.opt.step()

            if writer:
                writer.add_scalar('loss_emb', loss_emb, self.num_iter)
                writer.add_scalar('loss_ctr_U', loss_ctr_U, self.num_iter)
                writer.add_scalar('loss_ctr_q', loss_ctr_q, self.num_iter)
                writer.add_scalar('loss', loss, self.num_iter)

            self.num_iter += 1

            # -------- end bacth tasks
            
        # -------- end all tasks

        self.save_emb_net(log_dir+"/emb_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")
        self.save_ctr_net(log_dir+"/ctr_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")
        self.save_opt(log_dir+"/opt_epoch"+str(epoch)+"_"+f'{loss:.4f}'+".pt")

    def meta_test(self, task_loader, writer=None, ctr_scale=None):
        device = self.device
        # self.emb_net = torch.nn.DataParallel(self.emb_net, device_ids=[0,1,2,3])
        # self.ctr_net = torch.nn.DataParallel(self.ctr_net, device_ids=[0,1,2,3])

        for batch_task in tqdm(task_loader):

            batch_task_pre_emb, batch_task_post_emb, \
                batch_task_pre_ctr, batch_task_post_ctr = itertools.tee(batch_task, 4)

            # ----------------------------------------------------------------
            # make sentences
            # ----------------------------------------------------------------

            U_s = {} # support_sentence
            q_s = {} # query_sentence

            for task in batch_task_pre_emb:

                U_inps = task['train']['vision']
                U_inps = torch.cat([U_inps[:,0], U_inps[:,-1]], dim=1)
                q_inps = task['test']['vision']
                q_inps = torch.cat([q_inps[:,0], q_inps[:,-1]], dim=1)

                U_sj = self.emb_net(U_inps.to(device)).mean(0)
                U_sj = U_sj / torch.norm(U_sj)
                q_sj = self.emb_net(q_inps.to(device))[0]

                jdx = task['task_idx']
                U_s[jdx] = U_sj
                q_s[jdx] = q_sj

            # self.opt.zero_grad()

            # ----------------------------------------------------------------
            # calc loss_emb
            # ----------------------------------------------------------------

            loss_emb = 0

            for task in batch_task_post_emb:

                jdx = task['task_idx']
                U_sj = U_s[jdx]
                q_sj = q_s[jdx]

                for idx, U_si in list(U_s.items()):
                    if jdx == idx: continue

                    real = torch.dot(q_sj, U_sj)
                    fake = torch.dot(q_sj, U_si)
                    zero = torch.zeros(1).to(device)
                    loss_emb += torch.max(zero, 0.1 - real + fake)*1.0 # TODO:λ

            # loss_emb.backward(retain_graph=True) # memory saving

            # ----------------------------------------------------------------
            # prepare inputs for loss_ctr
            # ----------------------------------------------------------------

            U_sj_list = []
            q_sj_list = []

            for task in batch_task_pre_ctr:

                jdx = task['task_idx']
                U_sj_list.append(U_s[jdx])
                q_sj_list.append(q_s[jdx])

            # 64x20 -> 6400x20
            U_sj = torch.stack(U_sj_list).to(device).repeat_interleave(100, dim=0)
            q_sj = torch.stack(q_sj_list).to(device).repeat_interleave(100, dim=0)

            # ----------------------------------------------------------------
            # calc loss_ctr
            # ----------------------------------------------------------------

            U_vision, U_state, U_action, \
                q_vision, q_state, q_action = self.stack_demos(batch_task_post_ctr)

            U_state = torch.matmul(U_state, self.scale) + self.bias
            q_state = torch.matmul(q_state, self.scale) + self.bias

            output = self.ctr_net(U_vision, U_sj, U_state)
            loss_ctr_U = self.loss_fn(output, U_action)*len(U_vision)*0.1
            # loss_ctr_U.backward(retain_graph=True) # memory saving

            output = self.ctr_net(q_vision, U_sj, q_state)
            loss_ctr_q = self.loss_fn(output, q_action)*len(q_vision)*0.1
            # loss_ctr_q.backward() # memory saving

            loss = loss_emb.item() + loss_ctr_U.item() + loss_ctr_q.item()

            # self.opt.step()

            if writer:
                writer.add_scalar('val_loss_emb', loss_emb, self.num_iter)
                writer.add_scalar('val_loss_ctr_U', loss_ctr_U, self.num_iter)
                writer.add_scalar('val_loss_ctr_q', loss_ctr_q, self.num_iter)
                writer.add_scalar('val_loss', loss, self.num_iter)

            # self.num_iter += 1

            # -------- end bacth tasks

        # -------- end all tasks

    def make_test_sentence(self, demo_path, emb_net):
        device = self.device
        inp = (np.array(vread(demo_path).transpose(0,3,1,2)[[0,-1]].reshape(1,6,125,125)
                        , np.float32)-127.5)/127.5  # 1,6,125,125
        inp = torch.from_numpy(inp).to(device)
        inp = emb_net(inp.to(device))[0]
        return  inp / torch.norm(inp)
    
    def sim_test(self, env, model_path, demo_path, max_length=100):
        return super(TecNets, self).sim_test(env, model_path, demo_path, self.make_test_sentence, max_length=100)
