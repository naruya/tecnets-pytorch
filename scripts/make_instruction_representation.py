import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import pickle
import time

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

def _pre(X_train, X_train_99, y_train, X_test, X_test_99, y_test):
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    start_time = time.time()
    test_task = natsorted(glob.glob("/root/datasets/mil_sim_push/test/*"))
    # print(test_task)
    c = 0
    task_test_all = []
    for task in test_task:
        tasks_test = {}
        if task[-4:] == ".pkl": continue
        demos = natsorted(glob.glob(task + "/*"))
        with open(task + ".pkl", "rb") as f:
            task_name = pickle.load(f)["demo_selection"]
        task_name = task_name.split("/")[-1][:-4]
        # print(task_name)
        tasks_test["text"] = np.load("/root/share/tecnets-pytorch/datasets/2021_instructions/" + task_name + '.npy')
        tasks_test["images"] = []
        tasks_test["images_99"] = []
        for demo in demos:
            image = demo + "/0.gif"
            image_99 = demo + "/99.gif"
            # print(np.array(Image.open(image).convert('RGB')))
            tasks_test["images"].append(np.array(Image.open(image).convert('RGB')))
            tasks_test["images_99"].append(np.array(Image.open(image_99).convert('RGB')))
        tasks_test["images"] = np.array(tasks_test["images"])
        tasks_test["images_99"] = np.array(tasks_test["images_99"])
        assert tasks_test["images"].shape == (12, 125, 125, 3)
        assert tasks_test["images_99"].shape == (12, 125, 125, 3)
        assert tasks_test["text"].shape == (1, 128)
        # print(tasks.keys(), tasks["images"].shape, tasks["text"].shape)
        task_test_all.append(tasks_test)
    print(f"Process {len(task_test_all)} test tasks...")


    train_task = natsorted(glob.glob("/root/datasets/mil_sim_push/train/*"))
    # print(train_task)
    task_train_all = []
    for task in train_task:
        tasks_train = {}
        if task[-4:] == ".pkl": continue
        demos = natsorted(glob.glob(task + "/*"))
        with open(task + ".pkl", "rb") as f:
            task_name = pickle.load(f)["demo_selection"]
        task_name = task_name.split("/")[-1][:-4]
        # print(task_name)
        tasks_train["text"] = np.load("/root/share/tecnets-pytorch/datasets/2021_instructions/" + task_name + '.npy')
        tasks_train["images"] = []
        tasks_train["images_99"] = []
        for demo in demos:
            image = demo + "/0.gif"
            image_99 = demo + "/99.gif"
            # print(np.array(Image.open(image).convert('RGB')))
            tasks_train["images"].append(np.array(Image.open(image).convert('RGB')))
            tasks_train["images_99"].append(np.array(Image.open(image_99).convert('RGB')))
        tasks_train["images"] = np.array(tasks_train["images"])
        tasks_train["images_99"] = np.array(tasks_train["images_99"])
        assert tasks_train["images_99"].shape == (12, 125, 125, 3)
        assert tasks_train["images"].shape == (12, 125, 125, 3)
        assert tasks_train["text"].shape == (1, 128)
        # print(tasks.keys(), tasks["images"].shape, tasks["text"].shape)
        task_train_all.append(tasks_train)
    print(f"Process {len(task_train_all)} train tasks...")
    end_time = time.time()
    print(f"Pre-processing data cost {end_time - start_time} sec" )

    def get_data(X_train, X_train_99, y_train, X_test, X_test_99, y_test):
        """
        X_train = 769 * 12, 125, 125, 3
        y_train = 769 * 12, 128

        X_train = 74 * 12, 125, 125, 3
        y_train = 74 * 12, 128
        """
        for task in task_train_all:
            for i, image in enumerate(task["images"]):
                X_train.append(image)
                X_train_99.append(task["images_99"][i])
                y_train.append(task["text"])

        for task in task_test_all:
            for i, image in enumerate(task["images"]):
                X_test.append(image)
                X_test_99.append(task["images_99"][i])
                y_test.append(task["text"])
        return X_train, X_train_99, y_train, X_test, X_test_99, y_test

    start_time = time.time()
    print(f"Getting data...")
    
    X_train, X_train_99, y_train, X_test, X_test_99, y_test = get_data(X_train, X_train_99, y_train, X_test, X_test_99, y_test)

    # import ipdb; ipdb.set_trace()
    # language_shape == 128
    assert np.array(X_train).shape == (9228, 125, 125, 3)
    assert np.array(X_train_99).shape == (9228, 125, 125, 3)
    assert np.array(y_train).shape == (9228, 1, 128)
    assert np.array(X_test).shape == (888, 125, 125, 3)
    assert np.array(X_test_99).shape == (888, 125, 125, 3)
    assert np.array(y_test).shape == (888, 1, 128)

    end_time = time.time()
    print(f"Getting data cost {end_time - start_time} sec" )
    return X_train, X_train_99, y_train, X_test, X_test_99, y_test

X_train, X_train_99, y_train, X_test, X_test_99, y_test = [], [], [], [], [], []
X_train, X_train_99, y_train, X_test, X_test_99, y_test = _pre(X_train, X_train_99, y_train, X_test, X_test_99, y_test)

def save_data(X_train, X_train_99, y_train, X_test, X_test_99, y_test):
    X_train, X_train_99, y_train, X_test, X_test_99, y_test = np.array(X_train), np.array(X_train_99), np.array(y_train), np.array(X_test), np.array(X_test_99), np.array(y_test)
    print(X_train.shape)
    print(X_train_99.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(X_test_99.shape)
    print(y_test.shape)
    def save_pkl(file, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(file, f, protocol=4)
            print(f"Saved {file_name}!")
    save_pkl(X_train, "instruction_representation_X_train.pkl")
    save_pkl(X_train_99, "instruction_representation_X_train_99.pkl")
    save_pkl(y_train, "instruction_representation_y_train.pkl")
    save_pkl(X_test, "instruction_representation_X_test.pkl")
    save_pkl(X_test_99, "instruction_representation_X_test_99.pkl")
    save_pkl(y_test, "instruction_representation_y_test.pkl")
    print("Saved!")
save_data(X_train, X_train_99, y_train, X_test, X_test_99, y_test)