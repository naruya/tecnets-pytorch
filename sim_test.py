import argparse
import glob
import os
import imageio
import pickle
import numpy as np
import torch
import time
import moviepy.editor as mpy
from joblib import Parallel, delayed

from gym.envs.mujoco.pusher import PusherEnv
from tecnets import TecNets

XML_PATH = 'sim_push_xmls/'
CROP = False


def load_env(demo_info):
    xml_filepath = demo_info['xml']
    suffix = xml_filepath[xml_filepath.index('pusher'):]
    prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
    xml_filepath = str(prefix + suffix)

    env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
    return env


def load_demo(task_id, demo_dir, demo_inds):
    demo_info = pickle.load(open(demo_dir+task_id+'.pkl', 'rb'))
    demoX = demo_info['demoX'][demo_inds,:,:]
    demoU = demo_info['demoU'][demo_inds,:,:]
    d1, d2, _ = demoX.shape
    demoX = np.reshape(demoX, [1, d1*d2, -1])
    demoU = np.reshape(demoU, [1, d1*d2, -1])

    # read in demo video
    if CROP:
        demo_gifs = [imageio.mimread(demo_dir+'crop_object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]
    else:
        demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]

    return demoX, demoU, demo_gifs, demo_info


def eval_success(path):
    obs = path['observations']
    target = obs[:, -3:-1]
    obj = obs[:, -6:-4]
    dists = np.sum((target - obj) ** 2, 1)  # distances at each timestep
    return np.sum(dists < 0.017) >= 10


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task-Embedded Control Networks Implementation (meta-test)')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--demo_dir', type=str, default='/Users/tmats/workspace/tf_mil/mil/data/sim_push_test/')
    parser.add_argument('--log_dir', type=str, default='./test_log')
    parser.add_argument('--emb_model_path', type=str, default="./logs/emb.pt")
    parser.add_argument('--ctr_model_path', type=str, default="./logs/ctr.pt")
    parser.add_argument('--state_path', type=str, default="../mil/data/sim_push_common/scale_and_bias_sim_push.pkl")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"

    files = glob.glob(os.path.join(args.demo_dir, '*.pkl'))
    all_ids = [int(f.split('/')[-1][:-4]) for f in files]
    # all_ids = [int(f.split('/')[-1][6:-4]) for f in files]
    all_ids.sort()
    trials_per_task = 6

    gif_dir = args.log_dir + '/evaluated_gifs/'
    agent = TecNets(device=device)
    agent.sim_mode(args.emb_model_path, args.ctr_model_path, state_path=args.state_path)

    def rollout(input_tuple):
        ind, task_id = input_tuple
        demo_ind = 1  # for consistency of comparison
        demo_info = pickle.load(open(args.demo_dir + str(task_id) + '.pkl', 'rb'))
        # demo_info = pickle.load(open(args.demo_dir + "demos_" + str(task_id) + '.pkl', 'rb'))
        demo_path = args.demo_dir + 'object_' + str(task_id) + '/cond%d.samp0.gif' % demo_ind

        # load xml file
        env = load_env(demo_info)
        path = agent.sim_test(env, demo_path)
        video_filename = gif_dir + 'task_' + str(task_id) + '_' + str(ind % trials_per_task) + '.gif'
        clip = mpy.ImageSequenceClip([img for img in path['image_obs']], fps=20)
        clip.write_gif(video_filename, fps=20)
        env.close()
        sucess = eval_success(path)
        print(sucess)
        return sucess


    task_ids = [task_id for task_id in all_ids for _ in range(trials_per_task)]
    result = Parallel(n_jobs=args.num_workers)([delayed(rollout)(x) for x in enumerate(task_ids)])

    num_success = sum(result)
    num_trials = len(result)

    success_rate_msg = "Final success rate is %.5f" % (float(num_success) / num_trials)
    print(success_rate_msg)
