import argparse
import os
import glob
import numpy as np
import cv2
import moviepy.editor as mpy
from natsort import natsorted
from tqdm import tqdm

def vread(path, T=100):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif

def resize(path_from, path_to):
    os.mkdir(path_to)

    D0 = natsorted(glob.glob(os.path.join(path_from, "object_*"))) # 0~769

    for d0 in tqdm(D0): # 0~769
        D1 = natsorted(glob.glob(d0+"/*"))
        d0 = d0.split("/")[-1]
        os.mkdir(os.path.join(path_to, d0))

        for d1 in D1: # 0~24
            gif = vread(d1)
            gif = [cv2.resize(frame, (32,32)) for frame in gif]
            clip = mpy.ImageSequenceClip(gif, fps=20)
            d1 = d1.split("/")[-1]
            clip.write_gif(os.path.join(path_to, d0, d1), fps=20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mil_data', type=str, default='./mil_data')
    args = parser.parse_args()

    PATH = args.mil_data
    os.mkdir(os.path.join(PATH, "data_mini"))

    path_from = os.path.join(PATH, "data", "sim_push")
    path_to = os.path.join(PATH, "data_mini", "sim_push")
    resize(path_from, path_to)

    path_from = os.path.join(PATH, "data", "sim_push_test")
    path_to = os.path.join(PATH, "data_mini", "sim_push_test")
    resize(path_from, path_to)