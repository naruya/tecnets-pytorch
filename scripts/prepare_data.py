import glob

demo_dir='./datasets/mil_sim_push/'
task_paths = f'{demo_dir}train/task_*.pkl'
task_info_paths = glob.glob(task_paths)

num_support = 5
num_query = 1

print(task_info_paths)
for index in range(3):
    pickle_file = task_info_paths[index]
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print(data)