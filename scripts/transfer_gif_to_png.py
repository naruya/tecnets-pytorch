import glob
from PIL import Image
import tqdm
import datetime
import multiprocessing as mp
import os

demo_dir = '/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}new_test/task_*.pkl'
task_info_paths = glob.glob(task_paths)
print(len(task_info_paths))

def test(name, param):
    for index in param:
        if index >= len(task_info_paths): continue
        demo_path = task_info_paths[index][:-4] + f'/cond*/*.gif'
        demo_paths = glob.glob(demo_path)
        print(index)
        for demo in demo_paths:
            if os.path.exists(demo[:-4] + ".jpg"): continue
            Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("Local multi cpu : " + str(num_cores) + "cores")
    pool = mp.Pool(num_cores)
    
    param_dict = {}
    for i in range(40):
        param_dict[f"task{i}"] = list(range(i*2, (i+1)*2))

    results = [pool.apply_async(test, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]    
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Total cost time: " + "{:.2f}".format(elapsed_sec) + " sec")
