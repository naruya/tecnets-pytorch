import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import data, io
from PIL import Image
import pickle 

class Tecnetsdataset(Dataset):
    def __init__(self, demo_dir='./datasets/mil_sim_push/', train=True):
        # select the gif folder. 
        if train:
            task_paths = f'{demo_dir}train/task_*.pkl'
        else:
            task_paths = f'{demo_dir}test/task_*.pkl'     
        self.task_info_paths = glob.glob(task_paths) 

    def __len__(self):
        return len(self.task_info_paths)

    def __getitem__(self, index, num_support=5, num_query=1):
        # get data from task_paths.
        pickle_file = self.task_info_paths[index]     
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # import pdb; pdb.set_trace()
        # sample query, support from 6 ~ 18.
        num_sample = num_support + num_query
        support_query_sample_index = np.random.choice(12, num_sample, replace=False)

        # print('support_query_sample_index : ', support_query_sample_index)
        # test_sample_index = np.random.choice(list(set(range(12)) - set(support_query_sample_index)), num_sample, replace=False)
        # print('test_sample_index : ', test_sample_index)

        actions, states = [], [] # len(query + support), xx
        images = []
        for sample_index in support_query_sample_index:    
            demo_path = self.task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*'
            demo_paths = glob.glob(demo_path) 
            
            image = self._get_gif(demo_paths)
            # print(type(image))
            image = torch.stack(image)  # list to tensors.
            images.append(image) # list of tensors

            action = data['actions'][sample_index]
            actions.append(torch.from_numpy(action.astype(np.float32)).clone())

            state = data['states'][sample_index]
            states.append(torch.from_numpy(state.astype(np.float32)).clone())

        language_path = './datasets/2021_instructions/' + data['demo_selection'].split('/')[-1][:-4] + '.npy'
        language = np.load(language_path)

        # print(language.shape)
        # print(np.array(actions).shape)
        # print(np.array(images).shape)
        # print(np.array(states).shape)


        # import ipdb; ipdb.set_trace()
        # actions = np.array(actions)
        # print(type(actions))
        # actions = torch.from_numpy(actions.astype(np.float32)).clone()
        # print(type(actions), actions.shape)
        actions = torch.stack((actions))
        # print('actions_shape : ',actions.shape)  # torch.Size([6, 100, 7])
        states = torch.stack(states)
        # print('states_shape : ',states.shape)  # torch.Size([6, 100, 20])
        images = torch.stack(images)
        # print('images_shape : ', images.shape)  # torch.Size([6, 100, 125, 125, 3])
        language = torch.from_numpy(language.astype(np.float32)).clone()
        # print('langauge_shape : ', language.shape)  # torch.Size([1, 128])
        support_actions, query_actions = actions.split([num_support, num_query], dim=0)
        support_states, query_states = states.split([num_support, num_query], dim=0)
        support_images, query_images = images.split([num_support, num_query], dim=0)

        task_info = {
            'support_actions': support_actions,  # len(support), 100, 7.
            'support_states': support_states, # len(support), 100,20.
            'support_images': ((support_images.permute(0,1,4,2,3).to(torch.float32) - 127.5)/127.5),  # len(support), 100, 3, 125, 125.
            'support_languages': language, # len(support), 1, 128.
            'query_actions': query_actions,  # len(query), 100, 7.
            'query_states': query_states, # len(query), 100, 20.
            'query_images': ((query_images.permute(0,1,4,2,3).to(torch.float32) - 127.5)/127.5),  # len(query), 100, 3, 125, 125.
            'query_languages': language, # len(query), 1, 128.
        }
        return task_info
    
    def _get_gif(self, demo_paths):
        image = []
        for demo in demo_paths:
            x = Image.open(demo).convert('RGB')
            x = np.array(x)
            x = torch.from_numpy(x.astype(np.float32)).clone()
            image.append(x)
        return image
        
