import matplotlib.pyplot as plt
import distutils.util
import numpy as np


def image_show(image_bit_data):
    plt.figure()
    plt.imshow(image_bit_data, cmap='gray')
    plt.grid(False)
    plt.show()


# strサイズ 28*28
image_str = '0110000100001011111110010000110110011101111001100001001100000001101010001000111011000101010100010001000111111001110010100101111010100010001001101100111011101000110110000001010011110011100101011100010001111100101010100101100111110010110011110110110101110001011101110000110010111000000100001101101101111100110010100001100110110010111000001011001010100110111010010011011011000101011100001111110010001100010011111110000110110010100010110010111011100000110101110110001011010101010110110101000001100010010100010010011110111110110010111110011100110110111111101010011101100011100101100110000001101110111100110100100110111100001000010010000101101111001010101100101110010111010100110110111110101010001000001011010111111000100010010101001100101100101010010001001100011001100000100000100100010011'


# strをboolの2次元配列にデコード
image_bit_decode = np.array([[bool(distutils.util.strtobool(
    image_str[i + 28 * j])) for i in range(28)] for j in range(28)])
image_show(image_bit_decode)


# strをuint8の2次元配列にデコード
image_bit_decode = np.array([[bool(distutils.util.strtobool(
    image_str[i + 28 * j])) for i in range(28)] for j in range(28)], dtype='u8')
image_show(image_bit_decode)