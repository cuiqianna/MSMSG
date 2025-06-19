import numpy as np
from matplotlib import  pyplot as plt
import os
import cv2

def read_component(csv_file):
    with open(csv_file, mode='r', encoding= 'utf-8') as f:
        all = f.readlines()

    component =[]
    for i in all:
        i = i.strip().split(',')
        temp =[]
        for j in i:
            temp.append(int(j))
        component.append(temp)
    return component


def read_tau(csv_file):
    with open(csv_file, mode='r', encoding= 'utf-8') as f:
        all = f.readlines()

    component =[]
    for i in all:
        i = i.strip().split(',')
        temp =[]
        for j in i:
            temp.append(float(j))
        component.append(temp)
    tau = []
    num = len(component)
    for i in range(num):
        x = (component[i])[0]
        tau.append(x)
    return tau


def component_index(component, index):
    xx = []
    yy = []
    for co in component:
        x_set = np.where(index == co)[0]
        y_set = np.where(index == co)[1]
        xn = len(x_set)
        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]
            xx.append(dx)
            yy.append(dy)
    x_min = min(xx)
    x_max = max(xx) +1
    y_min = min(yy)
    y_max = max(yy) +1
    new_index = index[x_min:x_max, y_min: y_max]

    return x_min, x_max, y_min, y_max, new_index

def class_component(old_com):
    num = len(old_com)
    one_com = []
#    short_com =[]
    long_com = []
    for i in range(num):
        cur_com = old_com[i]
        cur_com_num = len(cur_com)
        if (cur_com_num == 1):
            one_com.append(i)
        else:
            # if (cur_com_num <= 10):
            #     short_com.append(i)
            # else:
            long_com.append(i)
    return one_com, long_com


def one_component_reidx(one_com,index, index_new,cur_num):
    index_new2 = index_new.copy()
    x_set = np.where(index == one_com)[0]
    y_set = np.where(index == one_com)[1]

    new_idx = cur_num +1
    xn = len(x_set)
    for i in range(xn):
        dx = x_set[i]
        dy = y_set[i]
        index_new2[dx,dy] = new_idx
    return index_new2

def reidx_com(cur_index, cur_num):
    hh = cur_index.shape[0]
    ww = cur_index.shape[1]
    new_idx = np.zeros(shape=(hh,ww), dtype=np.uint32)

    id_set = []
    for h in range(hh):
        for w in range(ww):
            id = cur_index[h,w]
            if (id not in id_set):
                id_set.append(id)
    id_set_new = [x for x in id_set if x != 0]
    num = len(id_set_new)
    for i in range(num):
        id = id_set_new[i]
        new_id = cur_num + i

        x_set = np.where(cur_index == id)[0]
        y_set = np.where(cur_index == id)[1]
        xn = len(x_set)
        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]
            new_idx[dx,dy] = new_id
    new_num = cur_num + num
    return new_idx, new_num



def new_image_show(image, index):
    hh = index.shape[0]
    ww = index.shape[1]


    id_max = np.max(index) +1

    image_new = np.zeros(shape=(hh,ww, 3), dtype= np.uint8)

    color = np.zeros(shape=(id_max, 3),dtype= np.float32)
    mean = np.zeros(shape=(id_max, 3),dtype= np.float32)
    for i in range( 1, id_max):
        x_set = np.where(index == i)[0]
        y_set = np.where(index == i)[1]
        xn = len(x_set)

        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]

            color[i,0] += image[dx, dy,0]
            color[i, 1] += image[dx, dy, 1]
            color[i, 2] += image[dx, dy, 2]
        if (xn != 0):
            mean[i,0] = color[i, 0] /xn
            mean[i, 1] = color[i, 1] / xn
            mean[i, 2] = color[i, 2] / xn
    for h in range(hh):
        for w in range(ww):
            id = index[h,w]
            image_new[h,w,0] = mean[id, 0]
            image_new[h,w,1] = mean[id,1]
            image_new[h,w,2] = mean[id, 2]

    im = image_new.copy()
    for h in range(1,hh-1):
        for w in range(1, ww-1):
            if (index[h,w] != index[h+1, w] or index[h,w] != index[h-1, w] or
            index[h,w] != index[h, w-1] or index[h,w] != index[h, w+1]):
                im[h,w,:] = (255, 0, 0)

    return im

def result_show_process(image, index, component, save_folder):
    height = image.shape[0]
    width = image.shape[1]
    index_new = np.zeros(shape=(height, width), dtype=np.uint32)

    img_new = image.copy()
    for h in range(height):
        for w in range(width):
            if (index[h, w] == 0):
                img_new[h, w, :] = 0

    com_read = component.copy()

    one_com, long_com = class_component(com_read)  # 保存的是com在component中的索引

    idx_num = 1
    merge_com_num = len(long_com)

    for k in range(merge_com_num):

        ii = long_com[k]  # 所有component中需要融合的com索引
        cur_com = com_read[ii]
        x_min, x_max, y_min, y_max, cur_index = component_index(cur_com, index)
        cur_height = cur_index.shape[0]
        cur_width = cur_index.shape[1]

        tem_com = '%d' % (ii + 1)
        cur_seg_name = os.path.join(save_folder, str(tem_com) + ".csv")
        seg_index = np.loadtxt(open(cur_seg_name), delimiter=',', dtype=np.uint32, skiprows=0)

        if (cur_height == 1 or cur_width == 1):
            seg_index2 = np.zeros(shape=(cur_height, cur_width), dtype=np.uint32)
            if cur_height == 1:
                for xx in range(cur_width):
                    seg_index2[0, xx] = seg_index[xx]
            if cur_width == 1:
                for xx in range(cur_height):
                    seg_index2[xx, 0] = seg_index[xx]
            new_seg_index, new_idx_num = reidx_com(seg_index2, idx_num)
            index_new[x_min: x_max, y_min: y_max] = new_seg_index.copy()

            idx_num = new_idx_num
        else:
            new_seg_index, new_idx_num = reidx_com(seg_index, idx_num)
            index_new[x_min: x_max, y_min: y_max] = new_seg_index.copy()

            idx_num = new_idx_num

    one_com_num = len(one_com)

    for c in range(one_com_num):
        xx = one_com[c]
        cur_comp = com_read[xx]
        x_min2, x_max2, y_min2, y_max2, cur_index2 = component_index(cur_comp, index)
        new_seg_index2, new_idx_num2 = reidx_com(cur_index2, idx_num)
        index_new[x_min2: x_max2, y_min2: y_max2] = new_seg_index2.copy()

        idx_num = new_idx_num2

    new_image = new_image_show(img_new, index_new)

    return index_new, new_image



