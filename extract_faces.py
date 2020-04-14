#!/usr/bin/python

import os
import cv2

# Extract faces from aligned facial images
# Author: Katarzyna Koptyra
# e-mail: kkoptyra@agh.edu.pl
# 2020

path = 'd:\\Projects\\python\\PycharmProjects\\img_align_celeba\\'
files = []

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

files.sort()

# znalezione drugim skrytpem
x = [54, 124]
y = [70, 179]

for i in range(len(files)):
    if i % 1000 == 0:
        print(str(i) + ' of ' + str(len(files)))
    img_help = cv2.imread(files[i])
    cv2.imwrite('d:\\Projects\\Python\\PycharmProjects\\face_recognition\\steganography\\gotowe\\only_faces\\%06d.png' % (i), img_help[y[0]:y[1],x[0]:x[1]])
