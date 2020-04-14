#!/usr/bin/python3

import EigenSteganographyLib as es
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import cv2
import os
import random
import DirectoryFunctions
from sklearn.metrics import mean_squared_error

# Evaluates prototype of steganography algorithm
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def generate_ellipse(size_x, size_y, size = 0.5):
    x_axis = np.linspace(-1, 1, size_x)[:, None]
    y_axis = np.linspace(-1, 1, size_y)[None, :]
    arr = x_axis ** 2 + y_axis ** 2
    arr = 1 - (arr / np.max(arr))
    #arr2 = np.copy(arr)
    if size >= 0:
        arr[arr >= 1-size] = 1
    else:
        arr[arr <= -size] = 0
    return arr

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


files = []

for r, d, f in os.walk('d:\\Projects\\Python\\PycharmProjects\\img_align_celeba\\'):
    for file1 in f:
        files.append(os.path.join(r, file1))
files.sort()

imgs = files[int(len(files) / 2):len(files)]

path = '.\\'
v_correct = np.load(path + "pca.res\\v_st_30375.npy")
w = np.load(path + "pca.res\\w_st_30375.npy")
mean_face = np.load(path + "pca.res\mean_face_st_30375.npy")
norms = np.load(path + "pca.res\\norms_st_30375.npy")
old_shape = np.load(path + "pca.res\\old_shape_st_30375.npy")
how_many_images = v_correct.shape[1]

# parameters
divider = 20

off = 1499
v_correct = v_correct[:, 0:4473]

proc = 0.22
quality = 92

vector_dl = [18, 37, 87, 174]
circle_sizes = np.arange(0.6, 0.9001, 0.1)


img_id = 0
vector_dl_id = 0
circle_size_id = 0

import time

start_start = time.time()

for img_id in range(len(imgs)):
    start = time.time()
    for vector_dl_id in range(len(vector_dl)):
        for circle_size_id in range(len(circle_sizes)):
            llen = vector_dl[vector_dl_id]
            circle_size = circle_sizes[circle_size_id]
            message_to_code = ''
            for a in range(0, llen):
                if a % 2 == 0:
                    message_to_code = message_to_code + 'a'
                else:
                    message_to_code = message_to_code + 'A'


            message = es.string2intarray(message_to_code)
            message = (message * 2 - 1) / divider

            x = [54, 124]
            y = [70, 179]

            img = cv2.imread(imgs[img_id], cv2.IMREAD_GRAYSCALE)
            img_cut = img.copy()
            img_cut = img_cut[y[0]:y[1],x[0]:x[1]]

            img_cut_compare = img_cut.copy()

            img_help_original = np.copy(img)


            encoded_data = es.encode(message, img_cut / 255, v_correct, mean_face, off)
            ell = generate_ellipse(y[1] - y[0], x[1] - x[0], circle_size)
            ell = ell * encoded_data + ((1-ell) * (img_help_original[y[0]:y[1],x[0]:x[1]] / 255))

            image_to_codeflat = img_cut_compare.flatten('F')
            elll = np.floor(np.clip(255 * ell, 0, 255).flatten('F'))

            mse = mean_squared_error(elll, image_to_codeflat)

            maxx = np.max(elll - image_to_codeflat)

            cc = correlation_coefficient(elll, image_to_codeflat)

            naz1 = imgs[img_id].split('\\')[-1].split('.')[0]

            DirectoryFunctions.append_line_to_file('./results/prototype1.csv',
                str(naz1) + ',' + 'clipping' + ',' + str(vector_dl[vector_dl_id]) + ',' +
                str(circle_sizes[circle_size_id]) + ',' + str(mse) + ',' + str(maxx)+ ',' + str(cc))

    end = time.time()
    print(str(img_id) + ' of ' + str(len(imgs)) + ' TIME: ' + str(end - start) + ' ALL TIME: ' + str(end - start_start))
