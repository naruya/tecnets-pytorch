import argparse
import glob
from PIL import Image
import tqdm
import datetime
import multiprocessing as mp
import os
import numpy as np

parser = argparse.ArgumentParser(description='transfer_git_to_xxx')
parser.add_argument('--task_type', type=str, default="train")
args = parser.parse_args()

demo_dir = '/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}{args.task_type}/task_*.pkl'
task_info_paths = glob.glob(task_paths)
print(len(task_info_paths))

to_npy = True


def test(name, param):
    for index in param:
        if index >= len(task_info_paths): continue
        demo_path = task_info_paths[index][:-4] + '/cond*/*.gif'
        demo_paths = glob.glob(demo_path)
        # for demo in demo_paths:
        #     if os.path.exists(demo[:-4] + ".jpg"): continue
        #     Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")
        if to_npy:
            demo_path = task_info_paths[index][:-4] + '/cond*'
            demo_paths = glob.glob(demo_path)
            for demo in demo_paths:
                if demo[-4:] == "*.npy": continue
                first_image_path = glob.glob(demo + "/0.gif")
                last_image_path = glob.glob(demo + "/99.gif")
                first_image = Image.open(first_image_path).convert('RGB')
                last_image = Image.open(last_image_path).convert('RGB')
                image_list = [np.array(first_image), np.array(last_image)]
                # np.save(first_image_path[:-6], np.array(image_list))
                print(first_image_path[:-6] + ".npy")
            # for demo in demo_paths:
            #     if demo[-6:] = "/0.gif":
            #         fast = Image.open(demo).convert('RGB')
            #     elif demo[-7:] = "/99.gif":
            #         last = Image.open(demo).convert('RGB')
                
            #     if os.path.exists(demo[:-4] + ".jpg"): continue
            #     Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")


            # print("last_demo: ", last_demo)
            # image_list = [np.array(Image.open(first_demo), np.float32), np.array(Image.open(last_demo), np.float32)]
            # np.save(first_demo[:-6], np.array(image_list))


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("Local multi cpu : " + str(num_cores) + "cores")
    pool = mp.Pool(num_cores)
    
    param_dict = {}
    num_task_each_worker_address = len(task_info_paths) // num_cores + 1
    for i in range(num_cores):
        param_dict[f"task{i}"] = list(range(i * num_task_each_worker_address, (i + 1) * num_task_each_worker_address))

    results = [pool.apply_async(test, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]    
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Total cost time: " + "{:.2f}".format(elapsed_sec) + " sec")


# demo_path = '/root/datasets/mil_sim_push/train/task_652/cond*/*.gif'
# demo_paths = glob.glob(demo_path)
# # print(index)
# for demo in demo_paths:
#     # if os.path.exists(demo[:-4] + ".jpg"): continue
#     Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")

