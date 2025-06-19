import numpy as np
import re_seg
import os
from matplotlib import  pyplot as plt
import MM_pca
import MM_reseg
from mean import mean_processing
from component import component_processing
from object_feature import object_feature_processing
from object_diff import object_diff_processing
from efficient_graph_merge import efficient_graph_processing
from result_show import result_show_process
from evaluation_merge import evaluation_merge_process
import math



import matplotlib as mpl
mpl.use('Agg')


def min_boundary_cal(index):
    num = np.max(index)+1
    boundary_set = []
    for i in range(num):
        x_set = np.where(index == i)[0]
        y_set = np.where(index == i)[1]
        x_min = np.min(x_set)
        x_max = np.max(x_set)
        y_min = np.min(y_set)
        y_max = np.max(y_set)
        x_dis = x_max - x_min +1
        y_dis = y_max - y_min + 1

        if (x_dis < y_dis):
            boundary_set.append(x_dis)
        else:
            boundary_set.append(y_dis)
    bound_min_val = math.ceil((min(boundary_set)) / 2)

    return bound_min_val


def component_save(component, save_file):
    with open(save_file, mode="w") as f:
        pass
    for i in component:
        # 每次拿一个list
        i = map(str, i)
        temp = ", ".join(i)
        with open(save_file, mode="a", encoding="utf-8") as f:
            f.write(temp + "\n")

def count_cal(index):
    height = index.shape[0]
    width = index.shape[1]

    s = np.max(index) + 1

    count = np.zeros(shape= (s), dtype= np.uint32)

    for h in range(height):
        for w in range(width):
            id = index[h,w]

            count[id] += 1
    return count


if __name__ == '__main__':
    #some load file has deleted, please replace it with your file name 
    image_file_name = os.path.join(file_folder,  "*.jpg")
    image = plt.imread(image_file_name)
    image_r = image[:,:,0]
    image_g = image[:,:,1]
    image_b = image[:,:,2]
    

    cur_seg = seg_num_set[0]#update with your data file
    tem_seg = '%d' % cur_seg



   

    cur_index_name = os.path.join(index_file_folder, str(tem_seg) + "_new.csv")
    print(cur_index_name)
    R_mean_name = os.path.join(index_file_folder, str(tem_seg) + "_Rmean.csv")
    G_mean_name = os.path.join(index_file_folder, str(tem_seg) + "_Gmean.csv")
    B_mean_name = os.path.join(index_file_folder, str(tem_seg) + "_Bmean.csv")
    count_name = os.path.join(index_file_folder, str(tem_seg) + "_count.csv")

    ori_seg = np.loadtxt(open(cur_index_name, "rb"), delimiter=',', dtype = np.uint32, skiprows=0)
    count = np.loadtxt(open(count_name, "rb"), delimiter=',', dtype = np.uint32, skiprows=0)

    r_mean = np.loadtxt(open(R_mean_name, "rb"), delimiter=',', dtype = np.float32, skiprows=0)
    g_mean = np.loadtxt(open(G_mean_name, "rb"), delimiter=',', dtype=np.float32, skiprows=0)
    b_mean = np.loadtxt(open(B_mean_name, "rb"), delimiter=',', dtype=np.float32, skiprows=0)

    Otsu_index, Otsu_count, Otsu_time = re_seg.reseg_processing(image, image_r, r_mean, image_g, g_mean, image_b, b_mean, ori_seg, count)
    
    save_Otsu_index_name = os.path.join(save_folder, str(tem_seg) + "_Otsu_index.csv")
    np.savetxt(save_Otsu_index_name, Otsu_index, delimiter=',', fmt= '%d')

    GMM_image_folder = os.path.join(file_folder, "GMM")

    GMM_file_name = os.path.join(GMM_image_folder, "GMM_LAB.jpg")
    image_GMM = plt.imread(GMM_file_name)

    initial_size = 3
    size_increment = 2
   
    max_size = min_boundary_cal(Otsu_index)
    if (max_size< initial_size):
        max_size = initial_size
    else:
        pass
    MM_pca_file = os.path.join(save_folder, "*.csv")
    
    MM_pca_matrix = np.loadtxt(open(MM_pca_file), delimiter=',', dtype=np.float32, skiprows=0)
    

    #Otsu_count = count_cal(Otsu_index)
    MM_index = MM_reseg.MM_reseg_processing(Otsu_index, MM_pca_matrix, Otsu_count)

    new_index, component, com_time = component_processing(MM_index) #MM_index: remove the minimum object

    

    save_component_name = os.path.join(save_folder, str(tem_seg) + "_component.csv")
    component_save(component, save_component_name)

    adj, count_new, color_f, gray_f, rlbp_f = object_feature_processing(image, image_file_name, new_index)
    object_diff = object_diff_processing(adj, color_f, gray_f, rlbp_f)


    merge_time = efficient_graph_processing(image, new_index, component, adj, count_new, object_diff, save_folder)



















