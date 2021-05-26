# TensorFlow と tf.keras のインポート
import tensorflow as tf
import numpy


c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

print(c)
# tf.Tensor(
# [[1. 2.]
#  [3. 4.]], shape=(2, 2), dtype=float32)

print(d)
# tf.Tensor(
# [[1. 1.]
#  [0. 1.]], shape=(2, 2), dtype=float32)

print(e)
# tf.Tensor(
# [[1. 3.]
#  [3. 7.]], shape=(2, 2), dtype=float32)

proto_tensor = tf.make_tensor_proto(e)


print(tf.make_ndarray(proto_tensor).tolist())
# array([[1., 3.],
#        [3., 7.]], dtype=float32)

print(type(e.numpy()))
# <class 'numpy.ndarray'>
