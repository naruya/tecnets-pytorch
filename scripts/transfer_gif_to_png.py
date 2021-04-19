import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import data, io
from PIL import Image
import pickle
import time

demo_dir='/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)

def test():
    # print(task_info_paths)
    for index in range(30):
        # pickle_file = task_info_paths[index]
        # with open(pickle_file, 'rb') as f:
        #     data = pickle.load(f)
    # print(data)

        for sample_index in range(12):
            demo_path = task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*.gif'
            demo_paths = glob.glob(demo_path)
            # _get_gif
            image = []
#            print(demo_paths[0])
            for demo in demo_paths:
                x = Image.open(demo)
                x = x.convert('RGB')
                x.save(demo[:-4] + ".png")