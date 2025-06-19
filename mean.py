import numpy as np
import math
import os
from matplotlib import  pyplot as plt


def re_idx(index):

    height = index.shape[0]
    width = index.shape[1]

    idx_val = []
    for h in range(height):
        for w in range(width):
            cur_id = index[h,w]
            if (cur_id not in idx_val):
                idx_val.append(cur_id)

    num = len(idx_val)

    new_index = np.zeros(shape= (height, width), dtype= np.uint32)

    for i in range(num):
        new_id = i
        id = idx_val[i]

        x_set = np.where(index == id)[0]
        y_set = np.where(index == id)[1]

        xn = len(x_set)
        for k in range(xn):
            dx = x_set[k]
            dy = y_set[k]

            new_index[dx, dy] = new_id

    return new_index, num

def new_adj(index, num):
    adj_matrix = np.zeros(shape=(num, num), dtype=np.uint8)
    xh = [-1, -1, -1, 0, 0, 1, 1, 1]
    yh = [-1, 0, 1, -1, 1, -1, 0, 1]

    height = index.shape[0]
    width = index.shape[1]

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            cur_id = index[h, w]
            for i in range(8):
                dx = h + xh[i]
                dy = w + yh[i]
                cur_adj_id = index[dx, dy]
                if (cur_id != cur_adj_id):
                    adj_matrix[cur_id, cur_adj_id] = 1
    return adj_matrix

def mean_cal(index, R, G, B, num):
    height = index.shape[0]
    width = index.shape[1]

    s = num
    #所有块的总数
    R_mean = np.zeros(shape= (s), dtype= np.float32)
    G_mean = np.zeros(shape=(s), dtype=np.float32)
    B_mean = np.zeros(shape=(s), dtype=np.float32)

    count = np.zeros(shape= (s), dtype= np.uint32)

    for h in range(height):
        for w in range(width):
            id = index[h,w]

            R_mean[id] += R[h,w]
            G_mean[id] += G[h, w]
            B_mean[id] += B[h, w]
            count[id] += 1

    for i in range(s):
        R_mean[i] /= count[i]
        G_mean[i] /= count[i]
        B_mean[i] /= count[i]

    return R_mean, G_mean, B_mean, count


def mean_processing(image, index):
    new_idx, seg_num = re_idx(index)
    adj_matrix = new_adj(new_idx, seg_num)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    R_mean, G_mean, B_mean, count = mean_cal(new_idx, R, G, B, seg_num)

    return new_idx, R_mean, G_mean, B_mean, count



