import os
import glob
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset as Taskset
from skimage.io import imread
from skimage import data, io
from PIL import Image


class test_dataset(Taskset):
    def __init__(self, demo_dir='/root/datasets/mil_sim_push/', train=True):
        # select the gif folder. 
        if train:
            demo_paths = f'{demo_dir}train/task_*/cond*.samp0/*.gif'
            task_paths = f'{demo_dir}train/task_*.pkl'
        else:
            demo_paths = f'{demo_dir}test/task_*/cond*.samp0/*.gif'
            task_paths = f'{demo_dir}test/task_*.pkl'
        self.task_info_paths = glob.glob(task_paths) 
        # print(len(self.tasks))
        print(len(self.task_info_paths), self.task_info_paths[:10])
        # self.demo_paths = glob.glob(demo_paths)
        # print(len(self.demo_paths), self.demo_paths[:10])

    def __getitem__(self, index):
        # train_demos_path = f'{self.demo_dir}train/task_{str(idx)}/cond{12}.samp0/{str(j)}.dif'
        # train_demo = torch.load(train_demos_files[0])
        demo_path = self.task_info_paths[index][:-4] + '/cond*/0.gif'
        
        demo = glob.glob(demo_path)


        x = Image.open(self.demo_paths[index]).convert('RGB')
        x = np.array(x)
        print(x.shape)
        io.imshow(x)
        io.show()
        # print(len(x), len(x[0]))
        # print(np.max(x), np.min(x), np.mean(x), np.std(x))
        return x
    
    def test_getitem(self, index):
        return self.__getitem__(index)
        
    def __len__(self):
        return len(self.task_info_paths)

# demo_dir = 

test = test_dataset()
# print(len(test))
for i in range(1):
    a = test.test_getitem(i)
    # print(a)
