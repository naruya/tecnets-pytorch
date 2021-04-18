import os
import time
import glob
import itertools
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ctr_net import ControlNet
from emb_net import EmbeddingNet
from utils import load_scale_and_bias
import cv2

class MetaLearner(object):
    def __init__(self, device, lr):
        
        self.device = device

        self.emb_net = EmbeddingNet().to(device)
        self.ctr_net = ControlNet().to(device)
        # self.emb_net = torch.nn.DataParallel(self.emb_net, device_ids=[0])
        # self.ctr_net = torch.nn.DataParallel(self.ctr_net, device_ids=[0])

        params = list(self.emb_net.parameters()) + list(self.ctr_net.parameters())

        if lr:
            self.opt = Adam(params, lr=lr)

    def save_emb_net(self, model_path):
        torch.save(self.emb_net.state_dict(), model_path)

    def save_ctr_net(self, model_path):
        torch.save(self.ctr_net.state_dict(), model_path)

    def save_opt(self, model_path):
        torch.save(self.opt.state_dict(), model_path)

    def load_emb_net(self, model_path, device):
        self.emb_net.load_state_dict(torch.load(model_path, map_location=device))
    
    def load_ctr_net(self, model_path, device):
        self.ctr_net.load_state_dict(torch.load(model_path, map_location=device))

    def load_opt(self, model_path, device):
        self.opt.load_state_dict(torch.load(model_path, map_location=device))

    def resume(self, emb_path, ctr_path, opt_path, device):
        self.load_emb_net(emb_path, device)
        self.load_ctr_net(ctr_path, device)
        self.load_opt(opt_path, device)

    def loss_fn(self, output, action):
        return F.mse_loss(output, action)

    def meta_train(self, task_loader):
        pass

    def meta_valid(self, task_loader):
        pass
    
    def make_test_sentence(self, demo_path, emb_net):
        pass

    def sim_mode(self, emb_model_path, ctr_model_path, state_path):
        device = self.device

        self.load_emb_net(emb_model_path, device)
        self.load_ctr_net(ctr_model_path, device)
        self.scale, self.bias = load_scale_and_bias(state_path)
        self.scale = torch.from_numpy(np.array(self.scale, np.float32)).to(device)
        self.bias = torch.from_numpy(np.array(self.bias, np.float32)).to(device)

    def sim_test(self, env, demo_path, make_test_sentence, max_length=100):
        device = self.device

        sentence = make_test_sentence(demo_path, self.emb_net)

        observations = []
        actions = []
        rewards = []
        images = []
        image_obses = []
        nonimage_obses = []

        o = env.reset()

        # settings for vision control
        if 'viewer' in dir(env):
            viewer = env.viewer
            if viewer is None:
                env.render()
                viewer = env.viewer
        else:
            viewer = env.wrapped_env.wrapped_env.get_viewer()
        viewer.autoscale()
        # new viewpoint
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat[0] = 0.4 # more positive moves the dot left
        viewer.cam.lookat[1] = -0.1 # more positive moves the dot down
        viewer.cam.lookat[2] = 0.0
        viewer.cam.distance = 0.75
        viewer.cam.elevation = -50
        viewer.cam.azimuth = -90
        env.render()
        if 'get_current_image_obs' in dir(env):
            image_obs, nonimage_obs = env.get_current_image_obs()
        else:
            image_obs, nonimage_obs = env.wrapped_env.wrapped_env.get_current_image_obs()

        image_obses.append(image_obs)
        image_obs = (np.array(np.expand_dims(image_obs, 0).transpose(0, 3, 1, 2), np.float32)-127.5)/127.5
        image_obs = torch.from_numpy(image_obs).to(device)
        nonimage_obses.append(nonimage_obs)
        nonimage_obs = np.array(np.expand_dims(nonimage_obs, 0), np.float32)
        nonimage_obs = torch.from_numpy(nonimage_obs).to(device)
        nonimage_obs = torch.matmul(nonimage_obs, self.scale) + self.bias
        observations.append(np.squeeze(o))

        for _ in range(max_length):
            a = self.ctr_net(image_obs, sentence, nonimage_obs).cpu().data.numpy()
            o, r, done, env_info = env.step(a)

            image_obs, nonimage_obs = env.get_current_image_obs()

            if done:
                break

            image_obses.append(image_obs)
            image_obs = (np.array(np.expand_dims(image_obs, 0).transpose(0, 3, 1, 2), np.float32)-127.5)/127.5
            image_obs = torch.from_numpy(image_obs).to(device)

            observations.append(np.squeeze(o))
            nonimage_obses.append(nonimage_obs)
            nonimage_obs = np.array(np.expand_dims(nonimage_obs, 0), np.float32)
            nonimage_obs = torch.from_numpy(nonimage_obs).to(device)
            nonimage_obs = torch.matmul(nonimage_obs, self.scale) + self.bias
            rewards.append(r)
            actions.append(np.squeeze(a))

            env.render()

        return dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            image_obs=np.array(image_obses),
            nonimage_obs=np.array(nonimage_obses),
        )
