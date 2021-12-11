import cv2
import numpy as np
from PIL import Image
import math


def dct(input_img, rows, cols):
    output_img = np.zeros((rows, cols), complex)
    for u in range(0, rows):  # Moving along rows
        for v in range(0, cols):  # Moving along cols
            for x in range(0, rows):  # Evaluation loop
                for y in range(0, cols):  # Evaluation loop
                    cosx = math.cos(((2 * x + 1) * u * math.pi) / (2 * rows))
                    cosy = math.cos(((2 * y + 1) * v * math.pi) / (2 * rows))
                    output_img[u][v] += input_img[x][y] * cosx * cosy

            if u == v == 0:
                output_img[u][v] = output_img[u][v] / rows
            else:
                output_img[u][v] = (output_img[u][v] * 2) / rows
    return output_img


def dft(input_img, rows, cols):
    output_img = np.zeros((rows, cols), complex)
    for u in range(0, rows):  # Moving along rows
        for v in range(0, cols):  # Moving along cols
            for x in range(0, rows):  # Evaluation loop
                for y in range(0, cols):  # Evaluation loop
                    output_img[u][v] += input_img[x][y] * np.exp(-1j * 2 * math.pi * (u * x / rows + v * y / cols))
            output_img[u][v] = output_img[u][v] / (rows * cols)
    return output_img


def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N <= 2:
        return dft(x)
    else:
        e = fft(x[::2])
        o = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([e + terms[:int(N / 2)] * o,
                               e + terms[int(N / 2):] * o])


def get_binary(decical, bits):
    return (bin(decical).replace("0b", "")).zfill(bits)


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def walsh(tm):
    n = int(math.log2(tm.shape[0]))

    for ridx in range(tm.shape[1]):  # row
        row_bin = get_binary(ridx, n)
        row_bin = row_bin[::-1]
        for cidx in range(tm.shape[0]):  # col
            col_bin = get_binary(cidx, n)
            B = 1
            for i in range(n):  # bit represent
                B = B * (-1) ** (int(col_bin[i]) * int(row_bin[i]))

            tm[cidx, ridx] = B

    return tm / math.sqrt(tm.shape[0])
