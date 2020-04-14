#!/usr/bin/python3

import EigenSteganographyLib as es
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import cv2
import os
import random

# Evaluates algorithms robustness against JPEG compression
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

def salt_and_pepper(img, proc, pn, px, shape):
	imgx = img.copy()
	img_flat = imgx.flatten()
	for i in range(int(proc * 0.01 * pn)):
		img_flat[px[i]] = 255 * random.randint(0,1)  # 3 *
	return img_flat.reshape(shape)

def generate_ellipse(size_x, size_y):
    x_axis = np.linspace(-1, 1, size_x)[:, None]
    y_axis = np.linspace(-1, 1, size_y)[None, :]
    arr = (x_axis / 150) ** 2 + (y_axis / 150) ** 2
    arr = 1 - (arr / np.max(arr))
    arr[arr >= 0.5] = 1
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
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row
	
	return previous_row[-1]


def testy(img, imgname, metoda, parametr, off, dlugosc):
	img_cut = img.copy()
	encoded_data = es.encode(message, img_cut / 255, v_correct, mean_face, off)
	if metoda == "rotation":
		encoded_data = ndimage.rotate(encoded_data, 0.25 * parametr, reshape=False, mode = 'nearest')  # {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
	img_help = np.clip(255 * encoded_data, 0, 255)
	naz1 = imgname.split('/')[-1].split('.')[0]
	if metoda == "salt-and-pepper":
		shape = img_help.shape
		pn = shape[0] * shape[1]
		px = [x for x in range(pn)]
		random.shuffle(px)
		img_help = salt_and_pepper(img_help, 0.1 * parametr, pn, px, shape)
		encoded_data = img_help
	elif metoda == "JPEG":
		naz = 'testJPEG1.jpg'
		cv2.imwrite(naz, img_help, [cv2.IMWRITE_JPEG_QUALITY, parametr])
		encoded_data = cv2.imread(naz, cv2.IMREAD_GRAYSCALE) / 255.0
	decoded_message = es.decode(encoded_data, v_correct, mean_face, off, len(message))
	decoded_message = np.round(((decoded_message * divider + 1) / 2))
	decoded_message = np.clip(decoded_message, 0 , 1)
	a1 = decoded_message
	a2 = es.string2intarray(message_to_code)
	a_diff = a1 - a2
	distance = np.count_nonzero(a_diff) / len(a_diff)
	decoded_message2 = es.intarray2string(decoded_message)

	f.write("%s,%s,%d,%d,%d,%d,%f\n" % (naz1, metoda, parametr, dlugosc, int(message_to_code == decoded_message2), levenshtein(message_to_code, decoded_message2), distance))



v_correct = np.load("pca.res/v_st_30375.npy")
w = np.load("pca.res/w_st_30375.npy")
mean_face = np.load("pca.res/mean_face_st_30375.npy")
norms = np.load("pca.res/norms_st_30375.npy")
old_shape = np.load("pca.res/old_shape_st_30375.npy")
how_many_images = v_correct.shape[1]


divider = 20
off = 1499
v_correct = v_correct[:,0:4473]
proc = 0.22
quality = 92
dlugosci = (18, 37, 87, 174, 370)

x = [54, 124]
y = [70, 179]

files = []
how_many_images = 30375

# r=root, d=directories, f = files
for r, d, f in os.walk('./only_faces'):
	for file1 in f:
		files.append(os.path.join(r, file1))
files.sort()

imgs = files[int(len(files)/2):len(files)]

random.seed(1)

f = open('results/robustness_test3.csv', 'w')

import time

start_start = time.time()
for iii in range(len(imgs)):
	start = time.time()
	imgname = imgs[iii]
	img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)

	for dlugosc in dlugosci:
		message_to_code = ''
		for a in range(0, dlugosc):
			if a % 2 == 0:
				message_to_code = message_to_code + 'a'
			else:
				message_to_code = message_to_code + 'A'
		#encode message as bits in numpy array of int
		message = es.string2intarray(message_to_code)
		message = (message * 2 - 1) / divider

		for j in [95, 92, 89, 86, 83, 80, 77, 74, 71, 68]:
			testy(img, imgname, "JPEG", j, 1499, dlugosc)

	end = time.time()
	print(str(iii) + ' of ' + str(len(imgs)) + ' TIME: ' + str(end - start) + ' ALL TIME: ' + str(end - start_start))
f.close()


