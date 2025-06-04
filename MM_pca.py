import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from spectral import *
from skimage.morphology import disk
from skimage.morphology import reconstruction
from skimage.morphology import erosion
from skimage.morphology import dilation
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time


def superpixel_size_compute(index, count):

    count_num = len(count)

    #ob_diag = np.zeros(shape=(count_num), dtype=np.uint32)
    short_dis = np.zeros(shape=(count_num), dtype=np.uint32)

    for i in range(count_num):
        x_set = np.where(index == i)[0]
        y_set = np.where(index == i)[1]

        x_min = np.min(x_set)
        x_max = np.max(x_set)
        y_min = np.min(y_set)
        y_max = np.max(y_set)

        x_dis = x_max - x_min +1
        y_dis = y_max - y_min +1

        sh = min([x_dis, y_dis])
        short_dis[i] = sh

    sh_min = min(short_dis)

    return sh_min



def opening_by_reconstruction(image, se):
    image_e = erosion(image, se)
    obr = reconstruction(image_e, image, method='dilation')

    return (obr)


def closing_by_reconstruction(image, se):
    # fechamento por reconstrução
    image_d = dilation(image, se)
    cbr = reconstruction(image_d, image, method='erosion')

    return (cbr)


def close_pca(image, initial_size, se_size_increment, max_size):

    tam = initial_size

    img_r = image[:, :, 0]
    img_g = image[:, :, 1]
    img_b = image[:, :, 2]

    size_sum = int((max_size - initial_size) / se_size_increment) + 1
    size_sum *= 3

    height = image.shape[0]
    width = image.shape[1]

    close_matrix = np.zeros(shape=(height, width, size_sum), dtype=np.float32)

    while tam <= max_size:
        se = disk(tam)
        cur_r = int((tam - initial_size) / se_size_increment)
        cur_g = int((tam - initial_size) / se_size_increment) + 1
        cur_b = int((tam - initial_size) / se_size_increment) + 2

        close_r = closing_by_reconstruction(img_r, se)
        close_g = closing_by_reconstruction(img_g, se)
        close_b = closing_by_reconstruction(img_b, se)

        close_matrix[:, :, cur_r] = close_r.copy()
        close_matrix[:, :, cur_g] = close_g.copy()
        close_matrix[:, :, cur_b] = close_b.copy()
        tam += se_size_increment

        pca_result = PCA_reduce(close_matrix)
        return pca_result





def open_pca(image, initial_size, se_size_increment, max_size):
    img_r = image[:,:,0]
    img_g = image[:,:,1]
    img_b = image[:,:,2]

    tam = initial_size

    size_sum = int((max_size - initial_size) / se_size_increment) + 1
    size_sum *= 3


    height = img_r.shape[0]
    width = img_r.shape[1]

    open_matrix = np.zeros(shape= (height, width, size_sum), dtype= np.uint32)


    while tam <= max_size:
        se = disk(tam)
        cur_r = int((tam- initial_size) / se_size_increment)
        cur_g = int((tam- initial_size) / se_size_increment) + 1
        cur_b = int((tam- initial_size) / se_size_increment) + 2

        open_r = opening_by_reconstruction(img_r,se)

        open_g = opening_by_reconstruction(img_g, se)

        open_b = opening_by_reconstruction(img_b, se)

        open_matrix[:,:, cur_r] = open_r.copy()
        open_matrix[:,:, cur_g] = open_g.copy()
        open_matrix[:,:, cur_b] = open_b.copy()

        tam += se_size_increment
    pca_result = PCA_reduce(open_matrix)
    return pca_result

def PCA_reduce(matrix_old):
    height = matrix_old.shape[0]
    width = matrix_old.shape[1]
    channel = matrix_old.shape[2]
    num_sum = height * width

    matrix = np.zeros(shape=(num_sum, channel), dtype=np.uint32)
    for h in range(height):
        for w in range(width):
            id = h * width + w
            for i in range(channel):
                matrix[id, i] = matrix_old[h, w, i]

    sc = StandardScaler()
    pixels = sc.fit_transform(matrix)

    pca = PCA(n_components=1)
    pc = pca.fit_transform(pixels)
    # print(pc.shape)

    matrix_new = np.zeros(shape=(height, width), dtype=np.float32)
    for h in range(height):
        for w in range(width):
            id = h * width + w
            matrix_new[h, w] = pc[id]

    return matrix_new


def im_show(image, pca_matrix):
    height = image.shape[0]
    width = image.shape[1]

    image_new = image.copy()

    for h in range(height):
        for w in range(width):
            open_pca_val = pca_matrix[h, w]
            if (open_pca_val <= 0):
                image_new[h, w, :] = 0
    return image_new

def MM_pca_processing(image, initial_size, size_increment, max_size):
    start = time.time()
    open_pca_matrix = open_pca(image, initial_size, size_increment, max_size)
    open_image = im_show(image, open_pca_matrix)

    close_pca_matrix = close_pca(open_image, initial_size, size_increment, max_size)
    end = time.time()

    MM_t = end - start
    #return open_pca_matrix, MM_t
    return close_pca_matrix, MM_t





#
# if __name__== '__main__':
#
#     main_folder = r"./"
