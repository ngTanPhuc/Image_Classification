from ann.layer.FCLayer import FCLayer
from ann.layer.convolutional import Convolutional
from ann.layer.reshape import Reshape
from ann.layer.loss_funcs import CCE, CCE_derivative
from ann.layer.activation_funcs import Sigmoid, Softmax
from ann.model.network import train, predict
from preprocessing.image2matrix import img2matr
from preprocessing.dataset import TensorDataset
from preprocessing.dataloader import DataLoader
import os
import numpy as np
import random

Label_map = {
    "Horses": [[1.], [0.], [0.], [0.]],
    "Dogs": [[0.], [1.], [0.], [0.]],
    "Cats": [[0.], [0.], [1.], [0.]],
    "Chickens": [[0.], [0.], [0.], [1.]]
}


# Load the image into a TensorDataset
def load_n_preprocess_data(data_dir, img_size=(64, 64), num_sample_each_class=100):
    d_n_l = []

    for label, one_hot in Label_map.items():
        class_dir = os.path.join(data_dir, label)  # Concatenate the data_dir and the label. e.g. dataimages/Horses
        if not os.path.exists(class_dir):
            break

        # Add all images in the class folder to this variable
        files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))][:num_sample_each_class]

        # Convert each image into a matrix
        for file in files:
            file_path = os.path.join(class_dir, file)  # The path to the image
            img_matrix = img2matr(file_path, img_size)

            d_n_l.append((img_matrix, one_hot))

    return d_n_l


data_n_labels = load_n_preprocess_data("data_images")
random.shuffle(data_n_labels)  # Shuffle the pairs of data and label
x_train = [pair[0] for pair in data_n_labels]
y_train = [pair[1] for pair in data_n_labels]
# train_data = TensorDataset(x, y)

network = [
    Convolutional((3, 64, 64), 3, 8),
    Sigmoid(),
    Reshape((8, 62, 62), (8 * 62 * 62, 1)),  # 8 * 62 * 62 = 30,752
    FCLayer(30752, 128),
    Sigmoid(),
    FCLayer(128, 4),
    Softmax()
]

# train_loader = DataLoader(train_data, 32, True, False)

train(
    network,
    CCE,
    CCE_derivative,
    x_train,
    y_train,
    500,
    0.1
)
