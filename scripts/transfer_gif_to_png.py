import glob
from PIL import Image
import tqdm


demo_dir = '/root/datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)


def test():
    for index in tqdm(range(len(task_info_paths))):
        for sample_index in range(12):
            demo_path = task_info_paths[index][:-4] + f'/cond{sample_index + 6}*/*.gif'
            demo_paths = glob.glob(demo_path)
            # _get_gif
            # print(demo_paths[0])
            for demo in demo_paths:
                Image.open(demo).convert('RGB').save(demo[:-4] + ".jpg")


test()
