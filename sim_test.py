import argparse
import glob
from datetime import datetime
import os
import imageio
import pickle
import json
import numpy as np
import torch
import time
from joblib import Parallel, delayed

from gym.envs.mujoco.pusher import PusherEnv
from tecnets import TecNets
from utils import vwrite

# this load_env cannot load train tasks
# def load_env(demo_info):
#     xml_filepath = demo_info['xml']
#     suffix = xml_filepath[xml_filepath.index('pusher'):]
#     prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
#     xml_filepath = str(prefix + suffix)
#     env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
#     return env

def load_env(demo_info):
    xml_path = demo_info['xml']
    xml_filepath = 'sim_push_xmls/' + xml_path.split("/")[-1]
    env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
    return env


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
    parser.add_argument('--emb_path', type=str, default="./logs/emb.pt")
    parser.add_argument('--ctr_path', type=str, default="./logs/ctr.pt")
    parser.add_argument('--state_path', type=str, default="../mil/data/sim_push_common/scale_and_bias_sim_push.pkl")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--eval', action='store_true', help="excute extra test")
    parser.add_argument('--no_save_gif', action='store_true', help="don't save episode video")

    args = parser.parse_args()

    device = "cuda"

    if args.eval:
        print("-"*60)
        print("!!!EVAL MODE!!! NOT AN OFFICIAL SCORE!")
        print("-"*60)

    log_dir = "./test_log/" + datetime.now().strftime('%m%d-%H%M%S')
    print(log_dir)
    print(args.emb_path)
    print(args.ctr_path)

    os.makedirs(os.path.join(log_dir, "evaluated_gifs"), exist_ok=True)

    models = {'emb_path':args.emb_path, 'ctr_path':args.ctr_path}
    with open(os.path.join(log_dir, "models.json"), "w") as f:
        json.dump(models, f)

    files = glob.glob(os.path.join(args.demo_dir, '*.pkl'))
    all_ids = [int(f.split('/')[-1][:-4]) for f in files] # test_task
    # all_ids = [int(f.split('/')[-1][6:-4]) for f in files] # train_task
    all_ids.sort()

    if args.eval:
        trials_per_task = 6
    else:
        trials_per_task = 6 # test_task
        # trials_per_task = 1 # train_task

    gif_dir = log_dir + '/evaluated_gifs/'
    agent = TecNets(device=device)
    agent.sim_mode(args.emb_path, args.ctr_path, state_path=args.state_path)

    def rollout(input_tuple):
        ind, (task_id, trial_id) = input_tuple

        if args.eval:
            demo_ind = trial_id + 9 # 0~11: right pushing, 12~23: left pushing. use [9:15].
        else:
            demo_ind = 1  # for consistency of comparison

        demo_info = pickle.load(open(args.demo_dir + str(task_id) + '.pkl', 'rb')) # test_task
        # demo_info = pickle.load(open(args.demo_dir + "demos_" + str(task_id) + '.pkl', 'rb')) # train_task
        demo_path = args.demo_dir + 'object_' + str(task_id) + '/cond%d.samp0.gif' % demo_ind
        print(demo_path)
        env = load_env(demo_info)
        prefix = "{}_{}_".format(task_id, trial_id)
        path, sentence = agent.sim_test(env, demo_path, prefix)
        env.close()
        success = eval_success(path)
        print(success)

        if not args.no_save_gif:
            video_filename = os.path.join(gif_dir, 'task_' + str(task_id) + '_' + str(ind % trials_per_task) + '.gif')
            gif = [img for img in path['image_obs']]
            vwrite(video_filename, gif)
            result = {'task_id': task_id, 'demo_path': demo_path, 'episode_path': video_filename, 'sentence': sentence, 'success': success}
        else:
            result = {'task_id': task_id, 'demo_path': demo_path, 'sentence': sentence, 'success': success}
        return result

    task_ids = [(task_id, trial_id) for task_id in all_ids for trial_id in range(trials_per_task)][:111]
    results = Parallel(n_jobs=args.num_workers)([delayed(rollout)(x) for x in enumerate(task_ids)])

    with open(os.path.join(log_dir,"results.pkl"), "wb") as f:
        pickle.dump(results, f)

    results = [result['success'] for result in results]
    num_success = sum(results)
    num_trials = len(results)
    success_rate_msg = "Final success rate is %.5f" % (float(num_success) / num_trials)
    print(success_rate_msg)
