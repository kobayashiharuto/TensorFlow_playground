# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt


def show_image():
    plt.figure()
    plt.imshow(train_images[1])
    plt.colorbar()
    plt.grid(False)
    plt.show()


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), \
    (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape, train_labels.shape)
show_image()
