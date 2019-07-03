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

import moviepy.editor as mpy 

# faster than skvideo.io
def vread(path, T=100):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif


def vwrite(path, gif):
    sys.stdout = open(os.devnull, 'w')
    clip = mpy.ImageSequenceClip(gif, fps=20)
    clip.write_gif(path, fps=20)
    sys.stdout = sys.__stdout__


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


def make_cache(demo_dir):

    print("making cache")

    os.mkdir(os.path.join(demo_dir, "cache"))

    gif_dirs = natsorted(glob.glob(os.path.join(demo_dir, "object_*")))
    pkl_files = natsorted(glob.glob(os.path.join(demo_dir, "*.pkl")))
    n_tasks = len(gif_dirs)

    for i in tqdm(range(n_tasks)):

        os.mkdir(os.path.join(demo_dir, "cache", "task"+str(i)))

        gif_dir = gif_dirs[i]
        gifs = natsorted(glob.glob(os.path.join(gif_dir, "cond*")))
        vision = [vread(path) for path in np.array(gifs)] # n,100,H,W,3

        pkl_file = pkl_files[i]
        with open(pkl_file, 'rb') as f:
            demo = pickle.load(f)
        state = list(np.array(demo['demoX'], np.float32)) # n,100,20
        action = list(np.array(demo['demoU'], np.float32)) # n,100,7

        n_demos = len(vision)

        for j in range(n_demos):
            demo = {
                'vision': torch.from_numpy(vision[j]), # not np.float32 but np.uint8 (for memory saving)
                'state': torch.from_numpy(state[j]), # np.float32
                'action': torch.from_numpy(action[j]) # np.float32
            }
            torch.save(demo, os.path.join(demo_dir, "cache", "task"+str(i), "demo"+str(j)+".pt"))
