# TensorFlow と tf.keras のインポート
# import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels), \
    (test_images, test_labels) = fashion_mnist.load_data()


print(type(train_images))
print(type(train_images[0]))
print(type(train_images[0][0]))
# print(train_images[0].tolist())

print(type(train_labels))
print(type(train_labels[0]))
print(train_labels[0])
# print(train_labels.tolist())
print(train_images.shape)
print(train_images[0].shape)
print(train_labels.shape)
