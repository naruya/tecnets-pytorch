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

num_support = 5
num_query = 1


def test():
# print(task_info_paths)
    for index in range(30):
        pickle_file = task_info_paths[index]
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    # print(data)
        if index == 0: print(data)

        action = data['actions']
        print(type(action), action.shape)
        #actions.append(action)

        state = data['states']
        print(type(state), state.shape)
        # states.append(torch.from_numpy(state.astype(np.float32)).clone())

        language_path = './datasets/2021_instructions/' + \
            data['demo_selection'].split('/')[-1][:-4] + '.npy'
        language = np.load(language_path)

        print(type(language), language.shape)
        task_info = {
            'demo_selection': data['demo_selection'],
            'states': state,
            'actions': action,
            'instructions': language
        }

        # print(task_info)


test()
