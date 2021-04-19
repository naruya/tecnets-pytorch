import glob
from PIL import Image
import tqdm
import datetime
import multiprocessing as mp


demo_dir = '/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)


def test(name, param):
    for index in param:
        for sample_index in range(12):
            demo_path = task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*.gif'
            demo_paths = glob.glob(demo_path)
            # _get_gif
            print(demo_paths[0])
            for demo in demo_paths:
                Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("Local multi cpu : " + str(num_cores) + "cores")
    pool = mp.Pool(num_cores)
    param_dict = {'task1': list(range(0, 100)),
                  'task2': list(range(100, 200)),
                  'task3': list(range(200, 300)),
                  'task4': list(range(300, 400)),
                  'task5': list(range(400, 500)),
                  'task6': list(range(500, 600)),
                  'task8': list(range(600, 700))}
    results = [pool.apply_async(test, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]    
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Total cost time: " + "{:.2f}".format(elapsed_sec) + " sec")