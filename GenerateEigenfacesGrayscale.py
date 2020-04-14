import cv2
import numpy as np
import os
import math

# Calculates LED distances between covariance matrices
# Author: Tomasz Hachaj, Katarzyna Koptyra
# e-mail: tomekhachaj@o2.pl, kkoptyra@agh.edu.pl
# 2020

path = 'd:\\Projects\\Python\\PycharmProjects\\face_recognition\\steganography\\gotowe\\only_faces'
files = []
how_many_images = 30375
variance_explained = 0.95

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

number_of_files = len(files)
test_set_length = math.floor(number_of_files / 2)
how_many_reference_images = test_set_length

a = 0
offset = how_many_reference_images

img = cv2.imread(files[0 + offset], cv2.IMREAD_GRAYSCALE)
old_shape = img.shape
img_flat = img.flatten('F')

T = np.zeros((img_flat.shape[0], how_many_images))
for i in range(how_many_images):
    if i % 1000 == 0:
        print("\tLoading " + str(i) + " of " + str(how_many_images))
    img_help = cv2.imread(files[i + offset], cv2.IMREAD_GRAYSCALE)
    T[:,i] = img_help.flatten('F') / 255


print('Calculating mean face')
mean_face = T.mean(axis = 1)

for i in range(how_many_images):
    T[:,i] -= mean_face


print('Calculating covariance')
C = np.matmul(T.transpose(), T)
C = C / how_many_images

print('Calculating eigenfaces')
from scipy.linalg import eigh
w, v = eigh(C)
v_correct = np.matmul(T, v)

sort_indices = w.argsort()[::-1]
w = w[sort_indices]  # puttin the evalues in that order
v_correct = v_correct[:, sort_indices]


norms = np.linalg.norm(v_correct, axis=0)# find the norm of each eigenvector
v_correct = v_correct / norms

#change all eigenvectors to have first coordinate positive - optional
for i in range(v_correct.shape[1]):
    if v_correct[0, i] < 0:
        v_correct[:, i] = -1 * v_correct[:, i]


#save results
np.save("pca.res//T_st_" + str(how_many_images), T)
np.save("pca.res//v_st_" + str(how_many_images), v_correct)
np.save("pca.res//w_st_" + str(how_many_images), w)
np.save("pca.res//mean_face_st_" + str(how_many_images), mean_face)
np.save("pca.res//norms_st_" + str(how_many_images), norms)
np.save("pca.res//old_shape_st_" + str(how_many_images), np.asarray(old_shape))
