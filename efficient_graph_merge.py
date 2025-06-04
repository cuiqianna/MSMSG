import numpy as np
import math
import os
from matplotlib import  pyplot as plt
import time
#
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


def object_cal(object_diff, count, adj_idx):

    object = []
    num = len(count)
    for i in range(1, num):
        cur_count = count[i]
        object_adj = []
        adj_diff = []
        for j in range(1, num):
            if adj_idx[i,j] == 1:
                object_adj.append(j)
                diff_val = object_diff[i,j]
                adj_diff.append(diff_val)

        object.append((i,object_adj, adj_diff, cur_count))
    return object

def com_adj_cal(component, object):
    cur_ob = []
    for x in component:
        ob = object[x-1]
        cur_ob.append(ob)
    return cur_ob

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

def reseg_idx(seg_result, index, image):
    num = len(seg_result)
    height = index.shape[0]
    width = index.shape[1]
    new_index = np.zeros(shape= (height, width), dtype= np.uint32)
    new_image = image.copy()
    for i in range(num):
        new_id = 1 + i
        ri = seg_result[i]
        r_num = len(ri)
        for j in range(r_num):
            id = ri[j]
            x_set = np.where(index == id)[0]
            y_set = np.where(index == id)[1]

            xn = len(x_set)
            for k in range(xn):
                dx = x_set[k]
                dy = y_set[k]

                new_index[dx, dy] = new_id
    for h in range(height):
        for w in range(width):
            if (new_index[h,w] == 0):
                new_image[h,w, :] = 0
    return new_index, new_image

def compute_seg_count(seg_index):
    #计算当前融合后的总segments
    seg_id = []
    height = seg_index.shape[0]
    width = seg_index.shape[1]
    for h in range(height):
        for w in range(width):
            id = seg_index[h,w]

            if (id not in seg_id):
                seg_id.append(id)
    if (0 in seg_id):
        num = len(seg_id) -1
    else:
        num = len(seg_id)
    return num


def var_cal(image, com_index):
    hh = com_index.shape[0]
    ww = com_index.shape[1]

    id_max = np.max(com_index) + 1

    image_new = np.zeros(shape=(hh, ww, 3), dtype=np.uint8)

    color = np.zeros(shape=(id_max, 3), dtype=np.float32)
    mean = np.zeros(shape=(id_max, 3), dtype=np.float32)

    var_sum = np.zeros(shape=(id_max, 3), dtype=np.float32)
    var = np.zeros(shape=(id_max, 3), dtype=np.float32)

    for i in range(1, id_max):
        x_set = np.where(com_index == i)[0]
        y_set = np.where(com_index == i)[1]
        xn = len(x_set)

        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]

            color[i, 0] += image[dx, dy, 0]
            color[i, 1] += image[dx, dy, 1]
            color[i, 2] += image[dx, dy, 2]
        mean[i, 0] = color[i, 0] / xn
        mean[i, 1] = color[i, 1] / xn
        mean[i, 2] = color[i, 2] / xn

        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]

            dif_r = image[dx,dy,0] - mean[i,0]
            dif_g = image[dx, dy, 1] - mean[i, 1]
            dif_b = image[dx, dy, 2] - mean[i, 2]

            var_sum[i,0] += pow(dif_r, 2)
            var_sum[i,1] += pow(dif_g,2)
            var_sum[i,2] += pow(dif_b,2)
        var[i, 0] = var_sum[i,0] / xn
        var[i, 1] = var_sum[i, 1] / xn
        var[i, 2] = var_sum[i, 2] / xn

    r_sum = 0.
    g_sum =0.
    b_sum = 0.
    for i in range(1, id_max):
        r_sum += var[i,0]
        g_sum += var[i,1]
        b_sum += var[i,2]

    var_val = (r_sum + g_sum + b_sum) / 3

    return var_val



class graph_merging:
    def __init__(self, object,k):
        #self.time = tt
        self.k = k
        self.object = object
        self.result = []
        self.graph_processing()

    def graph_processing(self):
        self.Int = []
        adj_index = []
        diff_v = []
        num = len(self.object)
        self.object_index = []
        for i in range(num):

            id = (self.object[i])[0]
            self.object_index.append(id)
            adj_idx = (self.object[i])[1]

            adj_num = len(adj_idx)

            dif = (self.object[i])[2]
            cur_ob_num = (self.object[i])[3]

            for j in range(adj_num):
                adj_index.append([id,adj_idx[j]])
                diff_v.append(dif[j])


        a = np.sort(diff_v)
        b = np.argsort(diff_v)
        c = []
        for i in range(b.shape[0]):
            c.append(adj_index[b[i]])

        self.result.append(c[0])


        for i in range(1,b.shape[0]):
            r_num = len(self.result)
            for j in range(r_num):
                r_i = self.result[j]
                r_len = len(r_i)
                Int = []
                for r in r_i:
                    #id = (self.object[r])[0]
                    id = self.object_index.index(r)
                    id_adj = (self.object[id])[1]
                    cur_diff = (self.object[id])[2]
                    for rj in r_i:
                        if (rj in id_adj and rj != r):
                            r_a = id_adj.index(rj)

                            Int.append(cur_diff[r_a])
                if (Int == []):
                    Int2 = 0
                else:
                    Int2 = self.MInt(Int, r_len)
                self.Int.append(Int2)
            x_a = a[i]
            x_b = c[i]

            x1 = -1
            x2 = -1
            for j in range(r_num):
                r_i = self.result[j]
                r_len = len(r_i)
                for k in range(r_len):
                    if (r_i[k] == x_b[0]):
                        x1 = j

            for k in range(r_num):
                r_i = self.result[k]
                r_len = len(r_i)
                for k_2 in range(r_len):
                    if (r_i[k_2] == x_b[1]):
                        x2 = k

            if (x1 == -1 and x2 == -1):
                self.result.append(x_b)

            if (x1 != -1 and x2 == -1):
                x_s1 = self.Int[x1]
                if x_a <= x_s1:
                    self.result[x1].append(x_b[1])
                else:
                    self.result.append([x_b[1]])
            if (x1 == -1 and x2 != -1):
                    x_s2 = self.Int[x2]
                    if x_a <= x_s2:
                        self.result[x2].append(x_b[0])
                    else:
                        self.result.append([x_b[0]])

            if (x1 != -1 and x2 != -1):
                if (x1 == x2):
                    self.result = self.result
                elif (x1 != x2):
                    x_s1 = self.Int[x1]
                    x_s2 = self.Int[x2]
                    x_min = min(x_s1,x_s2)
                    #print(x_min)

                    if (x_a <= x_min):
                        r2_l = len(self.result[x2])
                        for rr in range(r2_l):
                            self.result[x1].append((self.result[x2])[rr])
                        self.result.remove(self.result[x2])
                    else:
                        self.result =self.result

            self.result = [x for x in self.result if x != []]

            #print(len(self.result))
            #print(len(self.Int))
            self.Int = []
        return self.result

    def MInt(self,Int, n):
        t = self.k / n
        b = [i + t for i in Int]
        x = max(b)

        return x


def efficient_graph_processing(image, index, component, adj_matrix, count, object_diff, save_folder):
    img_new = image.copy()
    height = image.shape[0]
    width = image.shape[1]

    for h in range(height):
        for w in range(width):
            if (index[h, w] == 0):
                img_new[h, w, :] = 0

    object_adj = object_cal(object_diff, count, adj_matrix)

    com_read = component.copy()
    one_com, long_com = class_component(com_read)#保存的是com在component中的索引
    merge_com_num = len(long_com)

    time_sum = 0.
    optimal_tau_set = [] # save all the tau_value of all object_index

    for k in range(merge_com_num):
        ii = long_com[k]  #所有component中需要融合的com索引
        cur_com = com_read[ii]
        #print(k)
        #print(cur_com)

        tem_com = '%d' % (ii +1)

        seg_num = []

        tau_set =[]
        tau = 0.02
        seg_num_sum = len(seg_num)

        tau_opti = tau

        cur_object = com_adj_cal(cur_com, object_adj)
        x_min, x_max, y_min, y_max, cur_index = component_index(cur_com, index)
        cur_im = img_new[x_min: x_max, y_min: y_max]

        cur_index_opt = cur_index.copy()
        opti_var = 99999999.

        while ((seg_num_sum < 15) and (1 not in seg_num)):

            tem_tau = '%.2f' % tau

            start_1 = time.time()
            seg_result = graph_merging(cur_object, tau)
            end_1 = time.time()

            t_1 = end_1- start_1
            time_sum += t_1

            result = seg_result.result
            cur_new_index, new_image = reseg_idx(result, cur_index, cur_im)
            cur_seg_num = compute_seg_count(cur_new_index)

            if (cur_seg_num not in seg_num):
                seg_num.append(cur_seg_num)
                seg_num_sum += 1
                cur_height = cur_index.shape[0]
                cur_width = cur_index.shape[1]

                if (cur_height == 1 or cur_width == 1):
                    cur_index_opt = cur_new_index.copy()
                else:
                    start_2 = time.time()
                    cur_var= var_cal(cur_im, cur_new_index)
                    if (cur_var < opti_var):
                        tau_opti = tau
                        cur_index_opt = cur_new_index.copy()
                        opti_var = cur_var
                    else:
                        pass
                    end_2 = time.time()
                    t_2 = end_2 - start_2
                    time_sum += t_2
            else:
                pass
            tau += 0.02
        optimal_tau_set.append(tau_opti)

        save_cur_index_name = os.path.join(save_folder,  str(tem_com) +  ".csv")
        np.savetxt(save_cur_index_name, cur_index_opt, delimiter=',', fmt='%d')
    save_tau_opt_set_name = os.path.join(save_folder, "tau.csv")
    np.savetxt(save_tau_opt_set_name, optimal_tau_set, delimiter=',', fmt = '%.2f')
    return time_sum



#
# if __name__== '__main__':
#
#     main_folder = r"./"

#
#             component_save_file = os.path.join(component_folder, str(tem_number) +  "_component.csv")
#             com_read = read_component(component_save_file)
#
#             one_com, long_com = class_component(com_read)#保存的是com在component中的索引
#             merge_com_num = len(long_com)
#
#             time_sum = 0
#
#             for k in range(merge_com_num):
#                 ii = long_com[k]  #所有component中需要融合的com索引
#                 cur_com = com_read[ii]
#                 #print(k)
#                 #print(cur_com)
#
#                 tem_com = '%d' % (ii +1)
#
#                 seg_num = []
#
#                 tau_set =[]
#                 tau = 0.02
#                 seg_num_sum = len(seg_num)
#
#
#                 while ((seg_num_sum < 15) and (1 not in seg_num)):
#                     cur_object = com_adj_cal(cur_com, object_adj)
#                     x_min, x_max, y_min, y_max, cur_index = component_index(cur_com, index)
#                     new_im = img_new[x_min: x_max, y_min: y_max]
#
#                     tem_tau = '%.2f' % tau
#
#                     start = time.time()
#                     seg_result = graph_merging(cur_object, tau)
#                     end = time.time()
#
#                     t = end- start
#                     time_sum += t
#
#                     result = seg_result.result
#                     new_index, new_image = reseg_idx(result, cur_index, new_im)
#                     cur_seg_num = compute_seg_count(new_index)
#                     if (cur_seg_num not in seg_num):
#                         seg_num.append(cur_seg_num)
#                         tau_set.append(tau)
#                         seg_num_sum += 1
#
#
#                         result_im = new_image.copy()
#                         im_height = result_im.shape[0]
#                         im_width = result_im.shape[1]
#                         for h in range(1, im_height - 1):
#                             for w in range(1, im_width - 1):
#                                 if (new_index[h, w] != new_index[h - 1, w] or new_index[h, w] != new_index[h + 1, w] or
#                                         new_index[h, w] != new_index[h, w - 1] or new_index[h, w] != new_index[
#                                             h, w + 1]):
#                                     result_im[h, w, 0] = 255
#                                     result_im[h, w, 1] = 0
#                                     result_im[h, w, 2] = 0
#

#                     else:
#                         pass
#                     tau += 0.02
#                     tem_tau = '%.2f' % tau
#                 tau_name = os.path.join(save_folder,  str(tem_com) + "_" + "tau.csv")

#
#
