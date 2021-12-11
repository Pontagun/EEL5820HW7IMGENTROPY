import cv2
import numpy as np
from PIL import Image

import math




def tf_completion(htf):
    q1 = np.rot90(htf, 2)
    q2 = np.rot90(htf, 1)

    q_upper = np.concatenate((q1, q2), axis=1)

    q3 = np.rot90(htf, 3)
    q4 = htf

    q_lower = np.concatenate((q3, q4), axis=1)

    return np.vstack((q_upper, q_lower))


def ilpf(shape, d0):
    h_size = int(shape / 2)
    htf = (h_size, h_size)
    htf = np.zeros(htf)

    for r in range(h_size):
        for c in range(h_size):
            if math.sqrt(c * c + r * r) <= d0:
                htf[r][c] = 1.

    htfn = tf_completion(htf)

    return htfn


def ihpf(shape, d0):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            if math.sqrt(c * c + r * r) > d0:
                htf[r][c] = 1.

    htfn = tf_completion(htf)

    return htfn


def blpf(shape, d0, n):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            duv = math.sqrt(c * c + r * r)
            duvdo = (duv / d0) ** 2 * n
            htf[r][c] = 1 / (1 + duvdo)

    htfn = tf_completion(htf)

    return htfn


def bhpf(shape, d0, n):
    h_size = int(shape / 2)
    s = (h_size, h_size)
    htf = np.zeros(s)

    for r in range(h_size):
        for c in range(h_size):
            if r == 0 and c == 0:
                duv = 1
            else:
                duv = math.sqrt(c * c + r * r)

            doduv = (d0 / duv) ** 2 * n

            htf[r][c] = 1 / (1 + doduv)
            htf[r][c] = 1 / (1 + doduv)

    htfn = tf_completion(htf)

    return htfn