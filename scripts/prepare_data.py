import torch
import numpy as np
import pickle

import glob
import tqdm
import datetime
import multiprocessing as mp



demo_dir='/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)
print("task_info_paths: ", len(task_info_paths), task_info_paths[0])

def test(name, param):
    for index in param:
        if index >= len(task_info_paths): continue
        print(index)
        pickle_file = task_info_paths[index]
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        if index == 0: print(data)

        action = data['actions']
        print(type(action), action.shape)
        #actions.append(action)
        actions = []
        actions.append(torch.from_numpy(action.astype(np.float32)))

        state = data['states']
        print(type(state), state.shape)
        states = []
        states.append(torch.from_numpy(state.astype(np.float32)))

        language_path = './datasets/2021_instructions/' + \
            data['demo_selection'].split('/')[-1][:-4] + '.npy'
        language = np.load(language_path)
        language = torch.from_numpy(language.astype(np.float32))

        print(type(language), language.shape)
        # task_info = {
        #     'demo_selection': data['demo_selection'],
        #     'states': state,
        #     'actions': action,
        #     'instructions': language
        # }


if __name__ == '__main__':
    n_step = 20
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("Local multi cpu : " + str(num_cores) + "cores")
    pool = mp.Pool(num_cores)
    param_dict = {}
    for i in range(40):
        param_dict[f"task{i}"] = list(range(i * n_step, (i+1) * n_step))

    results = [pool.apply_async(test, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in tqdm(results)]    
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Total cost time: " + "{:.2f}".format(elapsed_sec) + " sec")
