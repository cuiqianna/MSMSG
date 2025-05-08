import  numpy as np
from matplotlib import  pyplot as plt
from skimage import io, color
import os
import math
import time

from mean import mean_processing

def var_cal(R, R_mean, G, G_mean, B, B_mean,index, count):
    height = R.shape[0]
    width = R.shape[1]

    count_num = len(count)

    var_r = np.zeros(shape=(count_num), dtype= np.float32)
    var_g = np.zeros(shape=(count_num), dtype=np.float32)
    var_b = np.zeros(shape=(count_num), dtype=np.float32)



    for h in range(height):
        for w in range(width):
            id = index[h,w]
            var_r[id] += pow((R[h,w] - R_mean[id]), 2)
            var_g[id] += pow((G[h, w] - G_mean[id]), 2)
            var_b[id] += pow((B[h, w] - B_mean[id]), 2)


    for i in range(count_num):
        var_r[i] /= count[i]
        var_g[i] /= count[i]
        var_b[i] /= count[i]

    var_sum = np.zeros(shape= (count_num), dtype= np.float32)
    for i in range(count_num):
        var_sum[i] = var_r[i] + var_g[i] + var_b[i]

    return var_sum


def var_cal_Otsu(R, R_mean, G, G_mean, B, B_mean,index, count):
    height = R.shape[0]
    width = R.shape[1]

    count_num = len(count)

    var_r = np.zeros(shape=(count_num), dtype= np.float32)
    var_g = np.zeros(shape=(count_num), dtype=np.float32)
    var_b = np.zeros(shape=(count_num), dtype=np.float32)



    for h in range(height):
        for w in range(width):
            id = index[h,w]
            var_r[id] += pow((R[h,w] - R_mean[id]), 2)
            var_g[id] += pow((G[h, w] - G_mean[id]), 2)
            var_b[id] += pow((B[h, w] - B_mean[id]), 2)


    for i in range(count_num):
        var_r[i] /= count[i]
        var_g[i] /= count[i]
        var_b[i] /= count[i]

    var_sum = np.zeros(shape= (count_num), dtype= np.float32)
    for i in range(count_num):
        var_sum[i] = var_r[i] + var_g[i] + var_b[i]

    var_result = 0.
    for i in range(count_num):
        var_result += var_sum[i]

    return var_result

def reseg_need(var_sum, threshold, count_num):
    par_val = int(math.ceil(count_num * threshold))#计算需要重分割的数量
    var = np.array(var_sum)

    var_sort_id = np.argsort(var) #var值从小到大排序的索引

    reseg_index = []

    start = count_num - par_val
    for i in range(start,  count_num):
        #var值大的那些索引提取出来，即为要重分割的id

        reseg_index.append(var_sort_id[i])
    return reseg_index




class reseg(object):
    def __init__(self, img, index, seg, count):
        self.img = img
        self.index = index
        self.seg = seg
        self.count = count
        self.height = self.seg.shape[0]
        self.width = self.seg.shape[1]
        self.img_pro()
    def img_pro(self):
        self.rgb =  self.img
        R = self.rgb[:,:,0]
        G = self.rgb[:,:,1]
        B = self.rgb[:,:,2]
        Gray = 0.299*R + 0.587*G + 0.114*B
        self.Gray = Gray.astype(np.uint32)

    def re_cluster(self):
        self.reseg_num = len(self.index)
        his = np.zeros(shape= (self.reseg_num, 256), dtype = np.uint32)
        bins = np.arange(0,256)
        N = np.zeros(shape=(self.reseg_num), dtype=np.uint32)

        proxi = np.zeros(shape=(self.reseg_num, 256), dtype=np.float32)
        for i in range(self.reseg_num):
            # locals()['re_' + str(i)] = []
            id_i = self.index[i]
            N[i] = self.count[id_i]
            a = np.where(self.seg == id_i)[0]
            b = np.where(self.seg == id_i)[1]
            n_i = len(a)
            for j in range(n_i):
                # locals()['re_' + str(i)].append((a[j], b[j]))
                x = a[j]
                y = b[j]
                intensity = self.Gray[x,y]
                his[i, intensity] += 1

        for i in range(self.reseg_num):
            for k in range(256):
                proxi[i,k] = his[i,k] / N[i]
        # for i in range(self.reseg_num):
        #     pro_sum = sum(proxi[i,:])
        #     print(pro_sum)
        self.all_thred = np.zeros(shape = (self.reseg_num), dtype=np.uint32)
        for i in range(self.reseg_num):
            sigma_max = 0.
            thr_val = 0
            cur_P = proxi[i,:]
            for j in range(256):
                sig = self.Otsu(cur_P, j)
                if sig > sigma_max:
                    sigma_max = sig
                    thr_val = j
            self.all_thred[i] = thr_val

        self.new_idx = self.seg.copy()

        old_num = len(self.count)
        for i in range(self.reseg_num):
            id_i = self.index[i]
            a = np.where(self.seg == id_i)[0]
            b = np.where(self.seg == id_i)[1]
            n_i = len(a)
            thred_val = self.all_thred[i]
            new_i = old_num + i
            for j in range(n_i):
                # locals()['re_' + str(i)].append((a[j], b[j]))
                x = a[j]
                y = b[j]
                intensity = self.Gray[x, y]
                if intensity > thred_val:
                    self.new_idx[x,y] = new_i

        return self.new_idx



    def Otsu(self,X, k):
        x1 = 0
        C1 = 0
        for j in range(k+1):
            x1 += X[j]
            C1 += X[j] * j
        x2 = 1 - x1
        C2 = 0
        for j2 in range(k+1, 256):
            C2 +=X[j2] * j2
        if x1==0:
            m1 = 0
        else:
            m1 = (1/ x1) *C1
        if x2 == 0:
            m2 = 0
        else:
            m2 = (1 / x2) * C2
        sigma = x1 * x2 * pow(m1-m2, 2)
        return sigma

def im_show(image, new_index):
    height = image.shape[0]
    width = image.shape[1]

    result_im = image.copy()
    for h in range(1,height-1):
        for w in range(1, width-1):
            if (new_index[h, w] != new_index[h - 1, w] or new_index[h, w] != new_index[h + 1, w] or
                    new_index[h, w] != new_index[h, w - 1] or new_index[h, w] != new_index[h, w + 1]):
                result_im[h, w, 0] = 255
                result_im[h, w, 1] = 0
                result_im[h, w, 2] = 0
    return result_im

def reseg_processing(image, image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count):
    var_var = var_cal(image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count)
    count_num = len(count)

    t_sum = 0.

    #var_Otsu = np.zeros(shape=(5), dtype= np.float32)
    #var_ori = var_cal_Otsu(image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count)
    #var_Otsu[0] = var_ori

    # var_val = var_ori
    # seg_new = ori_seg.copy()
    # var_min_id = 0
    # count_new = count.copy()

    k = 5
        #tem = '%d' % k
    threshold = k / 100

    reseg_need_idx = reseg_need(var_var, threshold, count_num)  # 计算需要重新分割的索引

    start_1 = time.time()
    process = reseg(image, reseg_need_idx, ori_seg, count)
    new_id = process.re_cluster()
    end_1 = time.time()
    t_1 = end_1 - start_1
    t_sum += t_1
    new_idx, R_mean, G_mean, B_mean, new_count = mean_processing(image, new_id)

    #cur_var = var_cal_Otsu(image_r, R_mean, image_g, G_mean, image_b, B_mean, new_idx, new_count)
    #
    # if (cur_var < var_val):
    #     seg_new = new_idx.copy()
    #     var_min_id = k
    #     count_new = new_count.copy()
    #     var_val = cur_var
    # else:
    #     pass
    return new_idx, new_count,  t_sum



def reseg_processing_opt(image, image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count, opt_k):
    var_var = var_cal(image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count)
    count_num = len(count)
    threshold_opt = opt_k / 100

    reseg_need_idx = reseg_need(var_var, threshold_opt, count_num)  # 计算需要重新分割的索引

    process = reseg(image, reseg_need_idx, ori_seg, count)
    new_id = process.re_cluster()

    new_idx, R_mean, G_mean, B_mean, new_count = mean_processing(image, new_id)
    return new_idx, new_count




