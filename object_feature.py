import numpy as np
import os
from matplotlib import  pyplot as plt
import feature
import cv2
import matplotlib as mpl
mpl.use('Agg')


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


def image_reconstruct(image, image_gray, index_file):

    index = np.loadtxt(open(index_file), delimiter=',', skiprows=0)
    index = index.astype(np.uint32)

    new_gray = image_gray.copy()
    new_image = image.copy()
    x_set = np.where(index == 0)[0]
    y_set = np.where(index == 0)[1]
    xn = len(x_set)
    for i in range(xn):
        dx = x_set[i]
        dy = y_set[i]

        new_gray[dx, dy] = 0
        new_image[dx,dy,:] = 0
    return new_image, new_gray

def object_feature(image_file, index, rlbp_matix, count):
    image = cv2.imread(image_file)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    num = len(count)

    color_feature = np.zeros(shape= (num, 3), dtype= np.float32)
    gray_feature = np.zeros(shape= (num), dtype= np.float32)
    rlbp_feature = np.zeros(shape= (num, 10), dtype = np.float32)


    for i in range(1, num):
        x_set = np.where(index == i)[0]
        y_set = np.where(index == i)[1]
        xn = len(x_set)
        h_sum = 0
        s_sum = 0
        v_sum = 0
        gray_sum = 0
        x_min = min(x_set)
        x_max = max(x_set) + 1
        y_min = min(y_set)
        y_max = max(y_set) + 1
        cur_image = rlbp_matix[x_min: x_max, y_min: y_max]
        hh = cv2.calcHist([cur_image], [0], None, [10], [0, 10])
        rlbp_hist = cv2.normalize(hh, hh).flatten()
        rlbp_feature[i,:] = rlbp_hist.copy()

        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]
            h_sum += image_hsv[dx, dy,0]
            s_sum += image_hsv[dx, dy,1]
            v_sum += image_hsv[dx, dy,2]
            gray_sum += image_gray[dx,dy]

        color_feature[i, 0] = h_sum / count[i]
        color_feature[i,1] = s_sum / count[i]
        color_feature[i,2] = v_sum / count[i]
        gray_feature[i] = gray_sum / count[i]
    color_feature /= 215
    gray_feature /= 215
    return color_feature, gray_feature, rlbp_feature
#
def adj_matrix_cal(idx_matrix):
    idx_num = np.max(idx_matrix) + 1
    height = idx_matrix.shape[0]
    width = idx_matrix.shape[1]

    adj_matrix = np.zeros(shape=(idx_num, idx_num), dtype=np.uint8)
    xh = [-1, -1, -1, 0, 0, 1, 1, 1]
    yh = [-1, 0, 1, -1, 1, -1, 0, 1]

    for h in range(height):
        for w in range(width):
            for i in range(8):
                if (h + xh[i] >= 0 and w + yh[i] >= 0 and h + xh[i] < height and w + yh[i] < width and
                        idx_matrix[h, w] != idx_matrix[h + xh[i], w + yh[i]]):
                    adj_matrix[int(idx_matrix[h, w])][int(idx_matrix[h + xh[i], w + yh[i]])] = 1

    new_adj = np.zeros(shape=(idx_num, idx_num), dtype=np.uint8)

    for i in range(idx_num):
        for j in range(idx_num):
            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                new_adj[i, j] = 1
                new_adj[j, i] = 0
    return new_adj




def count_cal(index):
    count_num = np.max(index) +1
    count_matrix = np.zeros(shape= (count_num), dtype= np.uint32)

    for i in range(count_num):
        x_set = np.where(index == i)[0]
        y_set = np.where(index == i)[1]

        xn = len(x_set)
        count_matrix[i] = xn
    return count_matrix

def object_feature_processing(image, image_file_name, index):
    img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    count = count_cal(index)

    adj = adj_matrix_cal(index)

    lbp = feature.LBP()
    rlbp = lbp.lbp_revolve_uniform(img_gray)

    color_f, gray_f, rlbp_f = object_feature(image_file_name, index, rlbp, count)
    return adj, count, color_f, gray_f, rlbp_f
#
# if __name__== '__main__':
#     main_folder = r"./"
#     file_str = ["ArcGIS", 'WDMI', 'google']
#     seg_str_set = ['SLIC', 'LSC', 'SNIC', 'FSLIC', 'CMSuG']
#
#     for i in range(2,3):
#         cur_str = file_str[i]
#         print(cur_str)
#
#         file_folder = os.path.join(main_folder, cur_str)
#         image_file_name = os.path.join(file_folder, cur_str + ".jpg")
#         image = plt.imread(image_file_name)
#         img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
#         # height = image.shape[0]
#         # width = image.shape[1]
#
#         index_folder = os.path.join(file_folder, "pre_seg/TC/200")
#         save_folder = os.path.join(file_folder, "pre_seg/TC/200")
#
#
#         for j in range(1):
#             str_seg = seg_str_set[j]
#             #print(str_seg)
#
#             cur_number = 200
#             tem_number = '%d' % cur_number
#
#             index_file = os.path.join(index_folder, str(tem_number) + "_new.csv")
#             print(index_file)
#             index = np.loadtxt(open(index_file), delimiter=',', dtype=np.uint32, skiprows=0)
#             #print(np.max(index))
#
#             count = count_cal(index)
#
#             adj = adj_matrix_cal(index)
#
#             lbp = feature.LBP()
#             rlbp=lbp.lbp_revolve_uniform(img_gray)
#
#
#             color_f, gray_f, rlbp_f = object_feature(image_file_name, index, rlbp, count)
#
#             color_fname = os.path.join(save_folder,  str(tem_number) + "_color_feature.csv")
#             gray_fname = os.path.join(save_folder,  str(tem_number) +"_gray_feature.csv")
#
#             rlbp_fname = os.path.join(save_folder, str(tem_number) + "_rlbp_feature.csv")
#
#             count_name = os.path.join(save_folder, str(tem_number) + "_count.csv")
#             adj_name = os.path.join(save_folder, str(tem_number) + "_adj_new.csv")
#
#             np.savetxt(color_fname, color_f, delimiter= ',', fmt ='%.4f')
#             np.savetxt(gray_fname,gray_f, delimiter= ',', fmt = '%.4f')
#             np.savetxt(rlbp_fname, rlbp_f, delimiter= ',', fmt = '%.3f')
#             np.savetxt(count_name, count, delimiter=',', fmt = '%d')
#             np.savetxt(adj_name, adj, delimiter=',', fmt = '%d')
#
#
#
#
#
