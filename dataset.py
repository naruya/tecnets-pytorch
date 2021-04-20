import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import data, io
from PIL import Image
import pickle
import re


class TecnetsDataset(Dataset):
    def __init__(self, demo_dir='./datasets/mil_sim_push/', train=True):
        # select the gif folder.
        if train:
            task_paths = f'{demo_dir}train/taskinfo*.pkl'
        else:
            task_paths = f'{demo_dir}test/taskinfo*.pkl'
        self.task_info_paths = glob.glob(task_paths)
        # print(len(self.task_info_paths))

    def __len__(self):
        return len(self.task_info_paths)

    @profile
    def __getitem__(self, index, num_support=5, num_query=1):
        # get data from task_paths.
        pickle_file = self.task_info_paths[index]
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # import pdb; pdb.set_trace()
        # sample query, support from 6 ~ 18.
        num_sample = num_support + num_query
        support_query_sample_index = np.random.choice(
            12, num_sample, replace=False)
        # print('support_query_sample_index : ', support_query_sample_index)

        actions = []
        states = [] 
        images = []
        for sample_index in support_query_sample_index:
            # print(self.task_info_paths[index])
            pickle_file
            demo_folder = re.sub('info', '', pickle_file)
            demo_path = demo_folder[:-4] + f'/cond{sample_index + 6}*/*.jpg' # 12 demos.
            demo_paths = glob.glob(demo_path)

            # _get_gif()
            image = [torch.from_numpy(np.array(Image.open(demo))) for demo in demo_paths]
            image = torch.stack(image)  # list to tensors.
            images.append(image)  # list of tensors

            action = data['actions'][sample_index]
            actions.append(torch.from_numpy(action.astype(np.float32)))

            state = data['states'][sample_index]
            states.append(torch.from_numpy(state.astype(np.float32)).clone())

        images = torch.stack(images)
        actions = torch.stack(actions)
        states = torch.stack(states)
        instructions = torch.from_numpy(np.array(data['instructions']))

        support_actions, query_actions = actions.split(
            [num_support, num_query], dim=0)
        support_states, query_states = states.split(
            [num_support, num_query], dim=0)
        support_images, query_images = images.split(
            [num_support, num_query], dim=0)
        # print(support_images.shape, len(support_actions))
        
        task_info = {
            'support_actions': support_actions,  # len(support), 100, 7.
            'support_states': support_states,  # len(support), 100,20.
            'support_images': ((support_images.permute(0, 1, 4, 2, 3) - 127.5) / 127.5),
            'support_instructions': instructions,  # len(support), 1, 128.
            'query_actions': query_actions,  # len(query), 100, 7.
            'query_states': query_states,  # len(query), 100, 20.
            'query_images': ((query_images.permute(0, 1, 4, 2, 3) - 127.5) / 127.5),
            'query_instructions': instructions,  # len(query), 1, 128.
        }
        return task_info

    # def _get_gif(self, demo_paths):
    #     return [torch.from_numpy(np.array(Image.open(demo))) for demo in demo_paths]