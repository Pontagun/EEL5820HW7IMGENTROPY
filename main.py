import cv2
import numpy as np
from PIL import Image
import math

from scipy.fftpack import dct, idct
from sklearn.metrics import mean_squared_error

import filter as flt
import kernel as knl


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])

    norm_image = unorm_image
    return norm_image


def center_image(image):
    rows = image.shape[0]
    cols = image.shape[1]
    centered_image = np.zeros((rows, cols))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            centered_image[i, j] = image[i, j] * ((-1) ** ((i - 1) + (j - 1)))

    return centered_image


def fft(img):
    rows = img.shape[0]
    cols = img.shape[1]

    img = center_image(img)
    img_width = rows * cols

    flatten_image = np.zeros(shape=(1, img_width))
    flatten_image = img.flatten()
    fft_image = knl.fft(flatten_image)
    fft_image_2d = np.reshape(fft_image, (rows, cols))

    # fft_image_2d = abs(fft_image_2d)
    # fft_image_2d = pixel_log255(fft_image_2d)

    # h = flt.ilpf(rows, cols/2)
    # # #
    # fft_image_2d = fft_image_2d * h
    #
    fft_image_2d = np.fft.ifft2(fft_image_2d)
    #
    fft_image_2d = abs(fft_image_2d)
    fft_image_2d = np.rot90(fft_image_2d, -1)
    fft_image_2d = np.flip(fft_image_2d, 1)
    # norm_image = pixel_log255(fft_image_2d)

    # # im = Image.fromarray(fft_image_2d)

    return fft_image_2d


def walsh(img):
    m = int(img.shape[0])
    n = int(img.shape[1])
    w = knl.walsh(np.zeros((m, n)))
    trf_img = w.dot(img).dot(w)
    flt = np.zeros((img.shape[0], img.shape[0]))
    flt[:int(m / 2), :int(n / 2)] = 1
    trf_img = trf_img * flt

    trf_img = w.dot(trf_img).dot(w)

    return trf_img


def dct(img):
    m = int(img.shape[0])
    n = int(img.shape[1])
    trf_img = knl.dct(img, img.shape[0], img.shape[1])

    flt = np.zeros((img.shape[0], img.shape[0]))
    flt[:int(m / 2), :int(n / 2)] = 1
    trf_img = trf_img * flt

    # trf_img = abs(trf_img)
    # trf_img = pixel_log255(trf_img)

    trf_img = idct2(trf_img)
    trf_img = abs(trf_img)

    return trf_img


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


if __name__ == '__main__':

    path = r'b64.jpg'
    img_st = cv2.imread(path, 0)
    img = img_st
    # img = fft(img)
    # img = walsh(img)
    img = dct(img)

    mse = mean_squared_error(img_st.flatten(), img.flatten(), squared=False)

    print(mse)

    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("b512qwefted.jpg")
