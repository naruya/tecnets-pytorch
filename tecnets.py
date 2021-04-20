import torch
import cv2
import itertools
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from meta_learner import MetaLearner
from utils import vread

import time
from memory_profiler import profile

class TecNets(MetaLearner):
    def __init__(self, device, learning_rate=None):
        super(TecNets, self).__init__(device, learning_rate)

    def make_sentence(self, image, normalize):
        N, k, _F, _C, W, H = image.shape
        
        # N,k,100,3,H,W
        inp = torch.cat([image[:, :, 0], image[:, :, -1]], dim=2)  # N,k,6,H,W
        inp = inp.view(N * k, 6, H, W)
        
        # import ipdb; ipdb.set_trace()
        sentence = self.emb_net(
            inp.view(N * k, 6, H, W)).view(N, k, 20)     # N,k,20
        # print(sentence)
        if normalize:
            sentence = sentence.mean(1)  # N,20
            sentence = sentence / torch.norm(sentence, 1)
        else:
            sentence = sentence[:, 0]    # N,20

        return sentence

    def cos_hinge_loss(
            self,
            query_sentence_list,
            support_sentence_list,
            U_si_list):
        # print(len(query_sentence_list))
        query_sentence = torch.stack(query_sentence_list)  # 4032, 20
        support_sentence = torch.stack(support_sentence_list)  # 4032, 20
        U_si = torch.stack(U_si_list)  # 4032, 20

        real = (query_sentence * support_sentence).sum(1)  # 4032,
        fake = (query_sentence * U_si).sum(1)  # 4032,
        zero = torch.zeros(1).to(self.device)  # 1,
        # print("zero: ", zero.device)
        loss = torch.max(0.1 - real + fake, zero)  # 4032,
        return loss

    @profile
    def meta_train(
            self,
            task_loader,
            num_batch_tasks=64,
            num_load_tasks=4,
            epoch=1000,
            writer=None,
            train=True):

        device = self.device

        loss_emb_list, loss_ctr_U_list, loss_ctr_q_list, loss_list = [], [], [], []

        for i, tasks in enumerate(tqdm(task_loader)):

            if (i * num_load_tasks) % num_batch_tasks == 0:
                if train:
                    self.opt.zero_grad()
                loss_emb, loss_ctr_U, loss_ctr_q = 0, 0, 0
                support_sentence_list, query_sentence_list = [], []

            # N,support_num,100,3,125,125
            import ipdb; ipdb.set_trace() # to check the tensor on the cuda.
            support_image = tasks["support_images"].to(device)
            support_state = tasks["support_states"].to(device)  # N,support_num,100,20
            support_action = tasks["support_actions"].to(device)  # N,support_num,100,7
            support_instruction = tasks['support_instructions'].to(device)

            query_image = tasks["query_images"].to(device)  # N,query_num,100,3,125,125
            query_state = tasks["query_states"].to(device)    # N,query_num,100,20
            query_action = tasks["query_actions"].to(device)  # N,query_num,100,7
            query_instruction = tasks['query_instructions'].to(device)  # # len(query), 1, 128.

            support_num, query_num = len(support_image[1]), len(query_image[1])
            size = support_image.shape[4]  # 125 or 64
            # print(support_image.device)
            # print(query_action.device)
            # print(query_instruction)
            support_sentence = self.make_sentence(
                support_image, normalize=True)  # N,20
            query_sentence = self.make_sentence(
                query_image, normalize=False)  # N,20

            support_sentence_list.append(support_sentence)
            query_sentence_list.append(query_sentence)

            # ---- calc loss_ctr ----
            support_image = support_image.view(num_load_tasks * support_num * 100, 3, size, size)  # N * support_num * 100,C,H,W
            support_state = support_state.view(num_load_tasks * support_num * 100, 20)            # N * support_num * 100,20
            support_action = support_action.view(num_load_tasks * support_num * 100, 7)           # N * support_num * 100,7
            # print(support_action.device)
            query_image = query_image.view(num_load_tasks * query_num * 100, 3, size, size)  # N * query_num * 100,C,H,W
            query_state = query_state.view(num_load_tasks * query_num * 100, 20)            # N * query_num * 100,20
            query_action = query_action.view(num_load_tasks * query_num * 100, 7)           # N * query_num * 100,7
            
            support_sentence_U_inp = support_sentence.repeat_interleave(100 * support_num, dim=0)        # N * support_num * 100,20
            support_sentence_q_inp = support_sentence.repeat_interleave(100 * query_num, dim=0)        # N * query_num * 100,20
            assert support_sentence_q_inp.is_cuda
            
            U_out = self.ctr_net(support_image, support_sentence_U_inp, support_state)
            _loss_ctr_U = self.loss_fn(U_out, support_action) * len(support_image) * 0.1

            q_out = self.ctr_net(query_image, support_sentence_q_inp, query_state)
            _loss_ctr_q = self.loss_fn(q_out, query_action) * len(query_image) * 0.1

            if train:
                _loss_ctr_U.backward(retain_graph=True)  # memory saving
                _loss_ctr_q.backward(retain_graph=True)
            loss_ctr_U += _loss_ctr_U.item()
            loss_ctr_q += _loss_ctr_q.item()

            # ----

            if ((i + 1) * N) % B == 0:

                # don't convert into list. graph informations will be lost.
                # (and an error will occur)
                support_sentence_list = torch.cat(support_sentence_list, 0)  # N * (B/N),20 -> N,20
                query_sentence_list = torch.cat(query_sentence_list, 0)  # N * (B/N),20 -> N,20

                query_sentence_j_list, support_sentence_j_list, U_si_list = [], [], []

                # ---- calc loss_emb ----
                for jdx, (query_sentence_j, support_sentence_j) in enumerate(zip(query_sentence_list, support_sentence_list)):
                    for idx, U_si in enumerate(support_sentence_list):
                        if jdx == idx:
                            continue
                        query_sentence_j_list.append(query_sentence_j)
                        support_sentence_j_list.append(support_sentence_j)
                        U_si_list.append(U_si)

                _loss_emb = torch.sum(self.cos_hinge_loss(query_sentence_j_list, support_sentence_j_list, U_si_list)) * 1.0

                if train:
                    _loss_emb.backward()
                    self.opt.step()
                loss_emb = _loss_emb.item()

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

        print(
            "loss:",
            loss,
            ", loss_ctr_U:",
            loss_ctr_U,
            ", loss_ctr_q:",
            loss_ctr_q,
            ", loss_emb:",
            loss_emb)

    def meta_valid(self, task_loader, num_batch_tasks, num_load_tasks, epoch):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            self.meta_train(
                task_loader,
                num_batch_tasks,
                num_load_tasks,
                epoch,
                train=False)

    def make_test_sentence(self, demo_path, emb_net):
        inp = vread(demo_path)
        cv2.imshow("demo", inp[-1][:, :, ::-1])
        cv2.waitKey(10)
        inp = torch.stack([torch.from_numpy(inp).to("cuda")])  # 1,F,H,W,C
        inp = (
            inp.permute(
                0, 1, 4, 2, 3).to(
                torch.float32) - 127.5) / 127.5  # 1,F,C,H,W
        inp = self.make_sentence(inp, normalize=True)  # 20,
        return inp

    def sim_test(self, env, demo_path):
        with torch.no_grad():
            self.emb_net.eval()
            self.ctr_net.eval()
            return super(TecNets,self).sim_test(
                env,
                demo_path,
                self.make_test_sentence)
