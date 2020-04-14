import EigenSteganographyLib as es
import cv2
import numpy as np
import random

# Prototype of steganography algorithm that works on single facial image
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

path = 'd:\\Projects\\python\\PycharmProjects\\img_align_celeba\\000002.jpg'

img_help = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_help_original = np.copy(img_help)
x = [54, 124]
y = [70, 179]
img_cut = img_help[y[0]:y[1],x[0]:x[1]]
icc = np.copy(img_cut)
cv2.imwrite('test_cut.png', img_cut)

v_correct = np.load("pca.res/v_st_30375.npy")
w = np.load("pca.res/w_st_30375.npy")
mean_face = np.load("pca.res/mean_face_st_30375.npy")
norms = np.load("pca.res/norms_st_30375.npy")
old_shape = np.load("pca.res/old_shape_st_30375.npy")
how_many_images = v_correct.shape[1]


divider = 20
off = 1499
v_correct = v_correct[:,0:4473]
#message to code

message_to_code = 'QWERTYUIOPAqwertyuiopa.'

message_to_code = ''
#[18, 37, 87, 174, 370]
llen = 37
for a in range(0, llen):
    if a % 2 == 0:
        message_to_code = message_to_code + 'a'
    else:
        message_to_code = message_to_code + 'A'

#encode message as bits in numpy array of int
message = es.string2intarray(message_to_code)
message = (message * 2 - 1) / divider
encoded_data = es.encode(message, img_cut / 255, v_correct, mean_face, off)
encoded_data2 = np.clip(256 * encoded_data, 0, 255)
print(len(message))

def generate_ellipse(size_x, size_y, size = 0.8):
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


ell = generate_ellipse(y[1] - y[0], x[1] - x[0], 0.6)
ell = ell * encoded_data + ((1-ell) * (img_help_original[y[0]:y[1],x[0]:x[1]] / 255))
img_help[y[0]:y[1],x[0]:x[1]]  = np.clip(255 * ell, 0, 255)


shape = img_help.shape
pn = shape[0] * shape[1]

cv2.imwrite('caly_obrazek.png', img_help)

img_help_rr = cv2.imread('caly_obrazek.png', cv2.IMREAD_GRAYSCALE) / 255
encoded_data = img_help_rr[y[0]:y[1],x[0]:x[1]]
decoded_message = es.decode(encoded_data, v_correct, mean_face, off, len(message))
decoded_message = np.round(((decoded_message * divider + 1) / 2))
decoded_message = np.clip(decoded_message, 0 , 1)

print(decoded_message)
print(es.string2intarray(message_to_code))
#print results
print("Encoded message:")
print(message_to_code)
print("Decoded message:")
print(es.intarray2string(decoded_message))

cv2.imshow('Image with hidden data', cv2.imread('caly_obrazek.png', cv2.IMREAD_GRAYSCALE))

ttt1 = cv2.imread('caly_obrazek.png', cv2.IMREAD_GRAYSCALE) * 1.0
ttt2 = cv2.imread('d:\\Projects\\python\\PycharmProjects\\img_align_celeba\\000002.jpg', cv2.IMREAD_GRAYSCALE) * 1.0

np.abs(ttt1 - ttt2) / 255
cv2.imwrite('caly_obrazek.png', np.abs(ttt1 - ttt2) )
cv2.imshow('Difference between original image and image with hidden data', np.abs(ttt1 - ttt2) / 255)

cv2.waitKey(0)
cv2.destroyAllWindows()
