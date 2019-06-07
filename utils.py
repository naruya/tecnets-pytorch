import sys
import site
sys.path.insert(0,site.getsitepackages()[0])
import numpy as np

from natsort import natsorted
import glob
import pickle
import cv2
from tqdm import tqdm
import os
import torch

# skvideo.ioのvreadより速いよ。
def vread(path, T=100):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif


# Not used in main function.
def get_demos(path):
    demo_files = path
    demo_files = natsorted(glob.glob(demo_files + '/*pkl'))

    demos = {} # {i(==task_number): {'demoU': array, 'xml': 'hoge.xml','demoX': array}, i+1: { ... }}
    for i, demo_file in enumerate(demo_files):
        with open(demo_file, 'rb') as f:
            demos[i] = pickle.load(f)
    return demos


def load_scale_and_bias(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
        scale = data['scale']
        bias = data['bias']
    return scale, bias


def make_cache(demo_dir, state_path):

    print("making cache_normalized")

    os.mkdir(os.path.join(demo_dir, "cache_normalized"))

    scale, bias = load_scale_and_bias(state_path)
    scale = torch.from_numpy(np.array(scale, np.float32))
    bias = torch.from_numpy(np.array(bias, np.float32))

    gif_dirs = natsorted(glob.glob(os.path.join(demo_dir, "object_*")))
    pkl_files = natsorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
    n_tasks = len(gif_dirs)

    for i in tqdm(range(n_tasks)):

        os.mkdir(os.path.join(demo_dir, "cache_normalized", "task"+str(i)))

        gif_dir = gif_dirs[i]
        gifs = natsorted(glob.glob(os.path.join(gif_dir, "cond*")))
        vision = [vread(path) for path in np.array(gifs)] # n,100,125,125,3

        pkl_file = pkl_files[i]
        with open(pkl_file, 'rb') as f:
            demo = pickle.load(f)
        state = list(np.array(demo['demoX'], np.float32)) # n,100,20
        action = list(np.array(demo['demoU'], np.float32)) # n,100,7

        n_demos = len(vision)

        vision = [(torch.from_numpy(v).permute(0,3,1,2).to(torch.float32)-127.5)/127.5 for v in vision]
        state = [torch.matmul(torch.from_numpy(s), scale) + bias for s in state]
        action = [torch.from_numpy(a) for a in action]

        for j in range(n_demos):
            demo = {
                'vision': vision[j],
                'state': state[j],
                'action': action[j]
            }
            torch.save(demo, os.path.join(demo_dir, "cache_normalized", "task"+str(i), "demo"+str(j)+".pt"))
