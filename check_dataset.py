import os
import glob
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset as Taskset
from skimage.io import imread
from skimage import data, io
from PIL import Image
import pickle 

class test_dataset(Taskset):
    def __init__(self, demo_dir='/root/datasets/mil_sim_push/', train=True):
        # select the gif folder. 
        if train:
            task_paths = f'{demo_dir}train/task_*.pkl'
        else:
            task_paths = f'{demo_dir}test/task_*.pkl'
        
        self.task_info_paths = glob.glob(task_paths) 

    def __getitem__(self, index, num_support=5, num_query=1):
        # get data from task_paths.
        pickle_file = self.task_info_paths[index]     
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # sample query, support from 6 ~ 18.
        num_sample = num_support + num_query
        random_sample_index = np.random.choice(12, num_sample, replace=False)
        # print(random_sample_index)

        actions, states = [], [] # len(query + support), xx
        images = [] 
        for sample_index in random_sample_index:    
            demo_path = self.task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*'
            demo_paths = glob.glob(demo_path) 
            images.append(self._get_gif(demo_paths))
            actions.append(data['actions'][sample_index])
            states.append(data['states'][sample_index])

        language_path = '/root/datasets/2021_instructions/' + data['demo_selection'].split('/')[-1][:-4] + '.npy'
        language = np.load(language_path)
        # print(language.shape)
        # print(np.array(actions).shape)
        # print(np.array(images).shape)
        # print(np.array(states).shape)

        task_info = {
            'actions': actions,  # len(support + query), 100, 7.
            'states': states, # len(support + query), 100, 7.
            'images': images,  # len(support + query), 100, 125, 125, 3.
            'languages': language, # len(support + query), 100, 128.
        }
        return task_info
    
    def _get_gif(self, demo_paths):
        image = []
        for demo in demo_paths:
            x = Image.open(demo).convert('RGB')
            image.append(np.array(x))
        return image
        
    def __len__(self):
        return len(self.task_info_paths)