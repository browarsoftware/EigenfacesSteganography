import EigenSteganographyLib as es
import cv2
import numpy as np
import random
import os
import DirectoryFunctions

# Evaluates algorithms robustness against scaling
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

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
# r=root, d=directories, f = files
for r, d, f in os.walk('d:\\Projects\\Python\\PycharmProjects\\img_align_celeba\\'):
    for file1 in f:
        files.append(os.path.join(r, file1))
files.sort()

imgs = files[int(len(files) / 2):len(files)]

path = 'd:\\Projects\\python\\PycharmProjects\\img_align_celeba\\000002.jpg'

img_help = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_help_original = np.copy(img_help)
x = [54, 124]
y = [70, 179]
img_cut = img_help[y[0]:y[1],x[0]:x[1]]
icc = np.copy(img_cut)
cv2.imwrite('test_cut.png', img_cut)
#steganogaphy
v_correct = np.load("pca.res/v_st_30375.npy")
w = np.load("pca.res/w_st_30375.npy")
mean_face = np.load("pca.res/mean_face_st_30375.npy")
norms = np.load("pca.res/norms_st_30375.npy")
old_shape = np.load("pca.res/old_shape_st_30375.npy")
how_many_images = v_correct.shape[1]


#parameters
divider = 20
off = 1499
v_correct = v_correct[:,0:4473]




off = 1499
v_correct = v_correct[:, 0:4473]
vector_dl = [18, 37, 87, 174, 370]
parameters = [0.8, 0.85, 0.95, 1.05, 1.15, 1.2]

img_id = 0
vector_dl_id = 0
circle_size_id = 0

import time

start_start = time.time()




for img_id in range(len(imgs)):
    start = time.time()
    for vector_dl_id in range(len(vector_dl)):
        for par_id in range(len(parameters)):
            llen = vector_dl[vector_dl_id]
            parameter = parameters[par_id]
            message_to_code = ''
            for a in range(0, llen):
                if a % 2 == 0:
                    message_to_code = message_to_code + 'a'
                else:
                    message_to_code = message_to_code + 'A'
            message = es.string2intarray(message_to_code)
            message = (message * 2 - 1) / divider

            img = cv2.imread(imgs[img_id], cv2.IMREAD_GRAYSCALE)

            img_cut = img.copy()
            img_cut = img_cut[y[0]:y[1],x[0]:x[1]]
            img_help_original = np.copy(img)
            encoded_data = es.encode(message, img_cut / 255, v_correct, mean_face, off)
            encoded_data2 = np.clip(256 * encoded_data, 0, 255)

            old_shape = encoded_data2.shape
            ell2 = cv2.resize(encoded_data2, (int(encoded_data2.shape[1] * parameter), int(encoded_data2.shape[0] * parameter)), interpolation=cv2.INTER_CUBIC)
            ell2 = cv2.resize(ell2, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_CUBIC)

            encoded_data = ell2
            decoded_message = es.decode(encoded_data, v_correct, mean_face, off, len(message))
            decoded_message = np.round(((decoded_message * divider + 1) / 2))
            decoded_message = np.clip(decoded_message, 0 , 1)

            decoded_message2 = es.intarray2string(decoded_message)

            a1 = decoded_message
            a2 = es.string2intarray(message_to_code)

            m1 = es.intarray2string(decoded_message)
            m2 = message_to_code

            a_diff = a1 - a2
            distance = np.count_nonzero(a_diff) / len(a_diff)

            leve = levenshtein(m1, m2)
            eqal = int(m1 == m2)

            naz1 = imgs[img_id].split('\\')[-1].split('.')[0]
            DirectoryFunctions.append_line_to_file('./results/robustness_test2.csv',
                str(naz1) + ',' + 'scaling' + ',' + str(vector_dl[vector_dl_id]) + ',' +
                str(parameters[par_id]) + ',' + str(eqal) + ',' + str(leve) + ',' + str(distance))
    end = time.time()
    print(str(img_id) + ' of ' + str(len(imgs)) + ' TIME: ' + str(end - start) + ' ALL TIME: ' + str(end - start_start))