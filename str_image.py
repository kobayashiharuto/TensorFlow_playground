import random
import numpy as np
import distutils.util

import matplotlib.pyplot as plt


def image_show(image_bit_data):
    plt.figure()
    plt.imshow(image_bit_data, cmap='gray')
    plt.grid(False)
    plt.show()


# bit配列を生成
image_bit_int = [random.getrandbits(1) for _ in range(28*28)]
# 28*28 の配列に変換
image_bit_array = [[bool(image_bit_int[i * j])
                    for i in range(28)] for j in range(1, 29)]
image_show(image_bit_array)


# bit配列をstrにエンコード
image_str_array = [str(image_bit_int[i]) for i in range(28*28)]
image_str = ''.join(image_str_array)
print(image_str)

# strをboolの2次元配列にデコード
image_bit_decode = np.array([[bool(distutils.util.strtobool(image_str[i * j]))
                              for i in range(28)] for j in range(1, 29)])
image_show(image_bit_decode)

# strをuint8の2次元配列にデコード
image_bit_decode = np.array([[bool(distutils.util.strtobool(image_str[i * j]))
                              for i in range(28)] for j in range(1, 29)], dtype='u8')
image_show(image_bit_decode)
