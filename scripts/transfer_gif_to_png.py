import glob
from PIL import Image
import tqdm
import datetime
import multiprocessing as mp
import os

demo_dir = '/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)


def test(name, param):
    for index in param:
        for sample_index in range(12):
            if index >= len(task_info_paths): continue
            demo_path = task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*.gif'
            demo_paths = glob.glob(demo_path)
            # _get_gif
#            print(demo_paths[0])
            for demo in demo_paths:
                if os.path.exists(demo[:-4] + ".jpg"):
                    Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("Local multi cpu : " + str(num_cores) + "cores")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(0, 50)),
                  'task2': list(range(50, 100)),
                  'task3': list(range(100, 150)),
                  'task4': list(range(150, 200)),
                  'task5': list(range(200, 250)),
                  'task6': list(range(250, 300)),
                  'task8': list(range(300, 350)),
                  'task9': list(range(350, 400)),
                  'task10': list(range(400, 450)),
                  'task11': list(range(450, 500)),
                  'task12': list(range(500, 550)),
                  'task13': list(range(400, 500)),
                  'task14': list(range(500, 550)),
                  'task15': list(range(550, 600)),
                  'task16': list(range(600, 650)),
                  'task17': list(range(650, 700)),
                  'task18': list(range(700, 800))}
    results = [pool.apply_async(test, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]    
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Total cost time: " + "{:.2f}".format(elapsed_sec) + " sec")
