import scipy.linalg as sc
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import cv2
import random
import math

# Helper functions
# Author: Tomasz Hachaj, Katarzyna Koptyra
# e-mail: tomekhachaj@o2.pl, kkoptyra@agh.edu.pl
# 2020

#path to data
path = 'd:\\Projects\\Python\\PycharmProjects\\face_recognition\\steganography\\gotowe\\only_faces\\'
#path to save results
path_to_save = 'd:\\Projects\\Python\\PycharmProjects\\face_recognition\\steganography\\gotowe\\results\\'


def led(A, B):
    A1 = sc.logm(A)
    B1 = sc.logm(B)
    return np.linalg.norm(A1 - B1, ord='fro')


def scale(np1):
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2

def scale_and_reshape(np1, mf, old_shape):
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product



files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

number_of_files = len(files)
test_set_length = math.floor(number_of_files / 2)
steps_count = 50

variance_explained_array = (0.999)
variance_explained = 0.95
step_size = math.floor(test_set_length / steps_count)
validation_sizes = np.arange(test_set_length, number_of_files, step_size)

print('Loading test data')
how_many_reference_images = test_set_length



img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
old_shape = img.shape
img_flat = img.flatten('F')

T = np.zeros((img_flat.shape[0], how_many_reference_images))
for i in range(how_many_reference_images):
    if i % 1000 == 0:
        print("\tLoading " + str(i) + " of " + str(how_many_reference_images))
    img_help = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    T[:, i] = img_help.flatten('F') / 255

mean_face = T.mean(axis=1)

for i in range(how_many_reference_images):
    T[:, i] -= mean_face

T_ref = T
mean_face_ref = mean_face
C_ref = np.matmul(T_ref, T_ref.transpose())
C_ref = C_ref / how_many_reference_images


a = 0
start = validation_sizes[0]

f = open(path_to_save + "led.txt", "a+")
f.write("ID ValidationStart ValidationEnd ValidationCount VarianceExplained HowManyEigens LED\n")
f.close()


for a in range(len(validation_sizes) - 1):
    print('Loading validation data: ' + str(a))
    #print(a)
    end = validation_sizes[a + 1]

    offset = how_many_reference_images
    img = cv2.imread(files[0 + offset], cv2.IMREAD_GRAYSCALE)
    old_shape = img.shape
    img_flat = img.flatten('F')

    how_many_images = end - start
    T2 = np.zeros((img_flat.shape[0], end - start))


    for i in range(how_many_images):
        if i % 1000 == 0:
            print("\tLoading " + str(i) + " of " + str(how_many_images))
        img_help = cv2.imread(files[i + offset], cv2.IMREAD_GRAYSCALE)
        T2[:,i] = img_help.flatten('F') / 255

    mean_face = T2.mean(axis = 1)

    for i in range(how_many_images):
        T2[:,i] -= mean_face

    C = np.matmul(T2, T2.transpose())
    C = C / how_many_images


    print('Calculating eigenfaces')
    from scipy.linalg import eigh

    w, v = eigh(C)
    #v_correct = np.matmul(T, v)
    v_correct = v
    sort_indices = w.argsort()[::-1]
    w = w[sort_indices]  # puttin the evalues in that order
    v_correct = v_correct[:, sort_indices]
    norms = np.linalg.norm(v_correct, axis=0)  # find the norm of each eigenvector
    v_correct = v_correct / norms

    variance = 0
    cooef_number = 0
    w_percen = w / sum(w)

#    for cc in range(len(variance_explained_array)):
    variance_explained = 0.999#variance_explained_array[cc]
    while variance < variance_explained:
        variance += w_percen[cooef_number]
        cooef_number = cooef_number + 1

    how_many_eigen = cooef_number

    #how_many_eigen = cooef_number
    print("requires ", how_many_eigen, " components to get ", variance, " of variance.")

    startE = 0
    stopE = how_many_eigen
    v_correct_use = v_correct[:, startE:stopE]

    # change all eigenvectors to have first coordinate positive - optional
    for i in range(how_many_eigen):
        if v_correct_use[0, i] < 0:
            v_correct_use[:, i] = -1 * v_correct_use[:, i]

    w_correct_use = w[startE:stopE]

    print('Calculating LED')
    led_res = led(C_ref, C)
    f = open(path_to_save + "led.txt", "a+")
    f.write("%d %d %d %d %lf %d %lf\n" % (a, start, end, how_many_images, variance_explained, how_many_eigen,led_res))
    f.close()

    print('Calculating image similarity')

    f_mse_cc = open(path_to_save + "mse_cc_" + str(a) + "_" + str(variance_explained) + ".txt", "wt")
    f_mse_cc.write("MSE CC\n")

    for bb in range(how_many_reference_images):
        if bb % 10000 == 0:
            print("\tProcessing " + str(bb) + " images")
        image_to_code = T_ref[:,bb] + mean_face_ref

        image_to_code = image_to_code - mean_face
        result = np.matmul(v_correct_use.transpose(), image_to_code)
        reconstruct = np.matmul(v_correct_use, result)

        reconstruct2 = scale_and_reshape(reconstruct, mean_face, old_shape)
        image_to_code2 = scale_and_reshape(image_to_code, mean_face, old_shape)

        reconstructflat = reconstruct2.flatten('F')
        image_to_codeflat = image_to_code2.flatten('F')

        reconstructflat = np.floor(np.clip(255 * reconstructflat, 0, 255))
        image_to_codeflat = np.floor(np.clip(255 * image_to_codeflat, 0, 255))

        mse = mean_squared_error(reconstructflat, image_to_codeflat)
        cc = correlation_coefficient(reconstructflat, image_to_codeflat)

        f_mse_cc.write("%lf %lf\n" % (mse, cc))

    f_mse_cc.close()
