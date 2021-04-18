import glob
import pickle
import numpy as np


demo_dir='/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)

num_support = 5
num_query = 1

# print(task_info_paths)
for index in range(3):
    pickle_file = task_info_paths[index]
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
   # print(data)

    num_sample = num_support + num_query
    support_query_sample_index = np.random.choice(12, num_sample, replace=False)   
    
    actions, states = [], []  # len(query + support), xx
    images = []
    

    for sample_index in support_query_sample_index:
        demo_path = self.task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*'
        demo_paths = glob.glob(demo_path)
        image = self._get_gif(demo_paths)
        # print(type(image))
        image = torch.stack(image)  # list to tensors.
        images.append(image)  # list of tensors

        action = data['actions'][sample_index]
        actions.append(torch.from_numpy(action.astype(np.float32)).clone())

        state = data['states'][sample_index]
        states.append(torch.from_numpy(state.astype(np.float32)).clone())

        language_path = './datasets/2021_instructions/' + \
            data['demo_selection'].split('/')[-1][:-4] + '.npy'
        language = np.load(language_path)

