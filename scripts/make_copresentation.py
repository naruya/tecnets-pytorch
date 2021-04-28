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

def _pre(X_train, y_train, X_test, y_test):
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
        tasks_test["text"] = np.load("/root/share/tecnets-pytorch/datasets/2021_there_is_a_and_b/" + task_name + '.npy')
        tasks_test["images"] = []
        for demo in demos:
            image = demo + "/0.gif"
            # print(np.array(Image.open(image).convert('RGB')))
            tasks_test["images"].append(np.array(Image.open(image).convert('RGB')))
        tasks_test["images"] = np.array(tasks_test["images"])
        assert tasks_test["images"].shape == (12, 125, 125, 3)
        assert tasks_test["text"].shape == (1, 768)
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
        tasks_train["text"] = np.load("/root/share/tecnets-pytorch/datasets/2021_there_is_a_and_b/" + task_name + '.npy')
        tasks_train["images"] = []
        for demo in demos:
            image = demo + "/0.gif"
            # print(np.array(Image.open(image).convert('RGB')))
            tasks_train["images"].append(np.array(Image.open(image).convert('RGB')))
        tasks_train["images"] = np.array(tasks_train["images"])
        assert tasks_train["images"].shape == (12, 125, 125, 3)
        assert tasks_train["text"].shape == (1, 768)
        # print(tasks.keys(), tasks["images"].shape, tasks["text"].shape)
        task_train_all.append(tasks_train)
    print(f"Process {len(task_train_all)} train tasks...")
    end_time = time.time()
    print(f"Pre-processing data cost {end_time - start_time} sec" )

    def get_data(X_train, y_train, X_test, y_test):
        """
        X_train = 769 * 12, 125, 125, 3
        y_train = 769 * 12, 768

        X_train = 74 * 12, 125, 125, 3
        y_train = 74 * 12, 768
        """
        for task in task_train_all:
            for image in task["images"]:
                X_train.append(image)
                y_train.append(task["text"])

        for task in task_test_all:
            for image in task["images"]:
                X_test.append(image)
                y_test.append(task["text"])
        return X_train, y_train, X_test, y_test

    start_time = time.time()
    print(f"Getting data...")
    
    X_train, y_train, X_test, y_test = get_data(X_train, y_train, X_test, y_test)

    # import ipdb; ipdb.set_trace()
    # language_shape == 768
    assert np.array(X_train).shape == (9228, 125, 125, 3)
    assert np.array(y_train).shape == (9228, 1, 768)
    assert np.array(X_test).shape == (888, 125, 125, 3)
    assert np.array(y_test).shape == (888, 1, 768)

    end_time = time.time()
    print(f"Getting data cost {end_time - start_time} sec" )
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = [], [], [], []
X_train, y_train, X_test, y_test = _pre(X_train, y_train, X_test, y_test)

# def save_data(X_train, y_train, X_test, y_test):
#     X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # def save_pkl(file, file_name):
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(file, f)
    #         print("Done!")
    # pickle.dump(X_test, "X_test")
    # pickle.dump(y_train, "y_train")
    # pickle.dump(y_test, "y_test")
    # save_pkl(X_train, "X_train.pkl")
    # save_pkl(y_train, "y_train.pkl")
    # save_pkl(X_test, "X_test.pkl")
    # save_pkl(y_test, "y_test.pkl")
    # print("Saved!")
# save_data(X_train, y_train, X_test, y_test)

# Main process from here.
# import wandb
# from wandb.keras import WandbCallback

# run = wandb.init(project='Convautoencoder-1',
#                  config={  # and include hyperparameters and metadata
#                      "learning_rate": 0.005,
#                      "epochs": 10,
#                      "batch_size": 64
#                     #  "loss_function": "sparse_categorical_crossentropy",
#                     #  "architecture": "CNN",
#                     #  "dataset": "CIFAR-10"
#                  })

# config = wandb.config

# # for i in range(4:
# class ConvolutionalAutoencoder(Model):
#     """
#     layer = tf.layers.conv2d(layer, 16, 5, 2, 'same')
#     layer = tf.contrib.layers.layer_norm(layer, activation_fn="elu")
#     """
#     def __init__(self):
#         super(ConvolutionalAutoencoder, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             layers.Input(shape=(125, 125, 3)),
#             layers.Conv2D(16, (5, 5), activation='elu', padding='same', strides=2),
#             layers.Conv2D(16, (5, 5), activation='elu', padding='same', strides=2),
#             layers.Conv2D(16, (5, 5), activation='elu', padding='same', strides=2),
#             layers.Conv2D(16, (5, 5), activation='elu', padding='same', strides=2),
#             layers.Flatten(),
#             layers.Dense(20)])

#         self.decoder = tf.keras.Sequential([
#             layers.Dense(125 * 125 * 3),
#             layers.Reshape((-1, 125, 125, 3)),
#             layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='elu', padding='same'),
#             layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='elu', padding='same'),
#             layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='elu', padding='same'),
#             layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='elu', padding='same'),
#             layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# autoencoder = ConvolutionalAutoencoder()
# # autoencoder.summary()

# optimizer = tf.keras.optimizers.Adam(config.learning_rate)
# autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError())

# # autoencoder.fit(X_train, X_train,
# #                 epochs=config.epochs,
# #                 batch_size=config.batch_size,
# #                 shuffle=True,
# #                 validation_data=(X_test, X_test),
# #                 callbacks=[WandbCallback()])