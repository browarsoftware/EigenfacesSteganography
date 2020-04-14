import cv2
import numpy as np

# Implementation of eigenfaces steganography library
# Author: Tomasz Hachaj
# e-mail: tomekhachaj@o2.pl
# 2020

#convert string to bit representation in string
def string2bits(s=''):
    return [bin(ord(x))[2:].zfill(8) for x in s]


#convert string to bit representation in numpy array of int
def string2intarray(s=''):
    a1 = string2bits(s)
    a2 = np.empty(len(a1) * 8)
    #print(a2.shape)
    idx = 0
    for a in a1:
        for b in range(8):
            a2[idx] = a[b]
            idx += 1
    return a2


#convert bit representation in string to string
def bits2string(b=None):
    return ''.join([chr(int(x, 2)) for x in b])


#convert numpy array rerpesentation of bits to string
def intarray2string(ia):
    rl = []
    idx = 0
    for a in range((int)(ia.shape[0] / 8)):
        rl2 = []
        for b in range(8):
            rl2.append((int)(ia[idx]))
            idx += 1
        rl.append(''.join(map(str,rl2)))
    return bits2string(rl)


#scale image
def scale(np_i):
    np1 = np.copy(np_i)
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2


#scale and reshape image for visualization
def scale_and_reshape(np_i, mf, old_shape):
    np1 = np.copy(np_i)
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2


#encode using eigenfaces (steganography)
def encode(data_to_code, carrier_img_i, v, mean_face, message_offset):
    carrier_img = np.copy(carrier_img_i)
    old_shape = carrier_img.shape
    img_flat = carrier_img.flatten('F')
    img_flat -= mean_face
    # generate eigenfaces from carier
    result = np.matmul(v.transpose(), img_flat)
    result_message = result
    # store message in features vector
    result_message[message_offset:(message_offset + data_to_code.shape[0])] = data_to_code
    # reconstruct carrier image
    reconstruct_message = np.matmul(v, result_message)
    image_to_code2 = scale_and_reshape(reconstruct_message, mean_face, old_shape)
    return image_to_code2


#encode using eigenfaces (steganography) to file
def encode_to_file(data_to_code, carrier_file, output_file, v, mean_face, message_offset):
    img = cv2.imread(carrier_file, cv2.IMREAD_GRAYSCALE) / 255
    output_image = encode(data_to_code, img, v, mean_face, message_offset)
    # save carrier image
    output_image = np.clip(255 * output_image, 0, 255)
    cv2.imwrite(output_file, output_image)


#decode using eigenfaces (steganography)
def decode(carrier_img_i, v, mean_face, message_offset, message_length):
    carrier_img = np.copy(carrier_img_i)
    # decode data
    img_flat_decode = carrier_img.flatten('F')
    img_flat_decode -= mean_face
    # generate features vector of eigenfaces coefficients
    result_decode_message = np.matmul(v.transpose(), img_flat_decode)
    # decoded message
    return result_decode_message[message_offset:(message_offset + message_length)]


#decode using eigenfaces (steganography) from file
def decode_from_file(carrier_file, v, mean_face, message_offset, message_length):
    # read encoded emage
    img_decode = cv2.imread(carrier_file, cv2.IMREAD_GRAYSCALE) / 255
    return decode(img_decode, v, mean_face, message_offset, message_length)


def test():
    #read eigenfaces data
    v_correct = np.load("pca.res/v_st_10000.npy")
    w = np.load("pca.res/w_st_10000.npy")
    mean_face = np.load("pca.res/mean_face_st_10000.npy")
    norms = np.load("pca.res/norms_st_10000.npy")
    old_shape = np.load("pca.res/old_shape_st_10000.npy")
    how_many_images = v_correct.shape[1]

    #parameters
    divider = 25
    #offset of message in carrier
    off = 1000
    #message to code
    message_to_code = 'Ble ble ble. Ala ma kota i co z tego?'
    #load carrier image
    path = 'e:\\Projects\\python\\same_twarze\\037302.png'

    #encode message as bits in numpy array of int
    message = string2intarray(message_to_code)
    #scale to range [-1, 1] and downscale
    message = (message * 2 - 1) / divider

    encode_to_file(message, path, "test2.jpg", v_correct, mean_face, off)
    decoded_message = decode_from_file("test2.jpg", v_correct, mean_face, off, message.shape[0])
    #scale to [0, 1]
    decoded_message = np.round(((decoded_message * divider + 1) / 2))
    decoded_message = np.clip(decoded_message, 0 , 1)

    #print results
    print("Encoded message:")
    print(message_to_code)
    print("Decoded message:")
    print(intarray2string(decoded_message))

    #show carrier image and carrier image with hidden data
    cv2.imshow('carrier', cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    cv2.imshow('hidden data', cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

