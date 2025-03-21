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
def load_n_preprocess_data(data_dir, img_size=(64, 64), num_sample_each_class=200):
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
                                    # x_train                           # y_train
tensor_data = TensorDataset([pair[0] for pair in data_n_labels], [pair[1] for pair in data_n_labels])
train_data = DataLoader(tensor_data, 16,True, False)

network = [
    Convolutional((3, 64, 64), 3, 8),
    Sigmoid(),
    Reshape((8, 62, 62), (8 * 62 * 62, 1)),  # 8 * 62 * 62 = 30,752
    FCLayer(30752, 512),
    Sigmoid(),
    FCLayer(512, 128),
    Sigmoid(),
    FCLayer(128, 4),
    Softmax()
]

# train_loader = DataLoader(train_data, 32, True, False)

train(
    network,
    CCE,
    CCE_derivative,
    train_data,
    500,
    0.2
)

# =================================== TEST ===================================
print("*** BEGIN TEST ***")
def load_n_preprocess_test(data_dir, img_size=(64, 64), num_sample_each_class=25):
    d_n_l = []

    for label, one_hot in Label_map.items():
        class_dir = os.path.join(data_dir, label)  # Concatenate the data_dir and the label. e.g. dataimages/Horses
        if not os.path.exists(class_dir):
            break

        # Add all images in the class folder to this variable
        files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))][1000: 1000 + num_sample_each_class]

        # Convert each image into a matrix
        for file in files:
            file_path = os.path.join(class_dir, file)  # The path to the image
            img_matrix = img2matr(file_path, img_size)

            d_n_l.append((img_matrix, one_hot))

    return d_n_l

test = load_n_preprocess_test("data_images")
tensor_test = TensorDataset([pair[0] for pair in test], [pair[1] for pair in test])
test_data = DataLoader(tensor_test, 100,True, False)

correct = 0
total = 0
for batch in test_data:  # batch is of type Batch
    batch_data = batch.batch_data
    batch_labels = batch.batch_labels

    # Forward propagation
    output = predict(network, batch_data)

    # Convert predictions to class indices
    y_pred = np.argmax(output, axis=1)
    y_true = np.argmax(batch_labels, axis=1)

    # Count correct predictions
    correct += np.sum(y_pred == y_true)
    total += batch_labels.shape[0]

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")

