import numpy as np
from matplotlib import pyplot as plt
import  os
import time

def new_index_cal(index):
    idx_max = np.max(index)

    index_new = index.copy()

    index_max = np.max(index) +1

    # super_new = []
    #
    # for i in range(1, index_max):
    #     super_new.append(i)

    for i in range(1, index_max):
        #经过背景消除技术后，背景像素标签为0，非背景像素为其他值
        id = i
        x_set = np.where(index == id)[0]
        y_set = np.where(index == id)[1]
        xn = len(x_set)
        pixel_all = []
        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]
            pixel_all.append((dx, dy))

        adj_all = []
        all_pixel_index = []

        xx = [-1, -1, -1, 0, 0, 1, 1, 1]
        yy = [-1, 0, 1, -1, 1, -1, 0, 1]

        for k in range(xn):
            all_pixel_index.append(k)
            idk = pixel_all[k]
            idk_x = idk[0]
            idk_y = idk[1]
            adj_k = []
            for l in range(8):
                adj_x = idk_x + xx[l]
                adj_y = idk_y + yy[l]
                adj_id = (adj_x, adj_y)

                if (adj_id in pixel_all):
                    cur_index = pixel_all.index(adj_id)
                    adj_k.append(cur_index)
            adj_all.append(adj_k)

        component = []

        while all_pixel_index:
            com = []
            xm = all_pixel_index[0]
            com.append(xm)
            cur_adj = adj_all[xm]
            all_pixel_index.remove(all_pixel_index[0])

            while cur_adj:
                cur_adj = list(set(cur_adj))
                for x_adj in cur_adj:
                    if (x_adj in com):
                        cur_adj.remove(x_adj)
                    else:
                        com.append(x_adj)
                        all_pixel_index.remove(x_adj)
                        ca = adj_all[x_adj]

                        for adj1 in ca:
                            cur_adj.append(adj1)
            component.append(com)

        com_num = len(component)
        if (com_num == 1):
            pass
        else:
            for ii in range(1, com_num):
                co = component[ii]
                new_id = idx_max + ii
                for cur_ii in co:
                    cur_pix = pixel_all[cur_ii]
                    x1 = cur_pix[0]
                    y1 = cur_pix[1]

                    index_new[x1, y1] = new_id
                # super_new.append(new_id)
            add_val = com_num - 1
            idx_max += add_val

    return index_new

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

    return adj_matrix



def mini_object_removal(index):
    #去除小于10的孤立像素簇
    reid_index = new_index_cal(index)

    new_index = reid_index.copy()
    num = np.max(reid_index) + 1
    adj_matrix = adj_matrix_cal(reid_index)


    for id in range(1, num):
        x_set = np.where(reid_index == id)[0]
        y_set = np.where(reid_index == id)[1]

        adj_val = adj_matrix[id,:]
        an = sum(adj_val[:])

        xn = len(x_set)
        if (an == 1 and xn < 10):
            aav_index = np.where(adj_val == 1)
            vv = aav_index[0]

            for i in range(xn):
                dx = x_set[i]
                dy = y_set[i]

                new_index[dx, dy] = vv

    index_f = final_index_cal(new_index)
    #重新给像素的索引排序

    return index_f

def final_index_cal(index):
    height = index.shape[0]
    width = index.shape[1]

    index_new = np.zeros(shape= (height, width), dtype= np.uint32)
    all_id = []

    for h in range(height):
        for w in range(width):
            id = index[h,w]
            if (id not in all_id):
                all_id.append(id)
    all_id.sort()

    num = len(all_id)
    for i in range(1, num):
        id = all_id[i]
        # new_id = i
        x_set = np.where(index == id)[0]
        y_set = np.where(index == id)[1]

        xn = len(x_set)

        for j in range(xn):
            dx = x_set[j]
            dy = y_set[j]

            index_new[dx, dy] = i

    return index_new

def component_cal(index):

    adj_matrix = adj_matrix_cal(index)

    index_max = np.max(index)+1
    super_new =[]

    for i in range(1, index_max):
        super_new.append(i)

    super2 = super_new.copy()
    super_adj = []

    for id in super_new:
        adj = []
        for j in range(1, index_max):
            if adj_matrix[id, j] == 1:
                adj.append(j)

        super_adj.append(adj)


    component = []
    al = []

    while super2:
        cur_com = []
        cur_com_adj = []
        cur_com.append(super2[0])
        #
        xm = super2[0]
        ym = super_new.index(xm)
        adj_set = super_adj[ym]
        for x in adj_set:
            cur_com_adj.append(x)
        #
        #     #print(cur_com_adj)
        super2.remove(xm)
        al.append(xm)
        #
        while cur_com_adj:
            cur_com_adj = list(set(cur_com_adj))
            #
            for x_adj in cur_com_adj:
                #             # print(pixel_copy.index(x_adj))
                if (x_adj in al):
                    cur_com_adj.remove(x_adj)
                #                 # print(cur_com_adj)
                else:
                    cur_com.append(x_adj)
                    #
                    super2.remove(x_adj)
                    #
                    al.append(x_adj)

                    xc = super_new.index(x_adj)

                    ca = super_adj[xc]
                    for xx in ca:
                        cur_com_adj.append(xx)
        #
        #             print(cur_com_adj)
        #         print(cur_com)
        #
        component.append(cur_com)
    return component

def component_save(component, save_file):
    with open(save_file, mode="w") as f:
        pass
    for i in component:
        # 每次拿一个list
        i = map(str, i)
        temp = ", ".join(i)
        with open(save_file, mode="a", encoding="utf-8") as f:
            f.write(temp + "\n")



def sub_image_show(image, image_index, image_component, path, seg_str):
    nn = len(image_component)
    height = image.shape[0]
    width = image.shape[1]

    for i in range(nn):
        tem = '%d' % (i+1)
        cur_co = image_component[i]
        im_new = np.zeros(shape= (height, width, 3), dtype= np.uint8)
        xx = []
        yy = []

        for co in cur_co:
            x_set = np.where(image_index == co)[0]
            y_set = np.where(image_index == co)[1]

            len_x = len(x_set)
            for j in range(len_x):
                dx = x_set[j]
                dy = y_set[j]
                xx.append(dx)
                yy.append(dy)
                im_new[dx, dy,:] = image[dx,dy,:]
        x_min = min(xx)
        x_max = max(xx)+ 1
        y_min = min(yy)
        y_max = max(yy)+ 1
        new_im = im_new[x_min: x_max, y_min: y_max]

        #path = r"./sub_image"
        im_name = os.path.join(path, seg_str + "_sub_"+ str(tem) + ".jpg")
        plt.imsave(im_name, new_im)





def image_show(index, image):
    result_im = image.copy()

    height = index.shape[0]
    width = index.shape[1]

    for h in range(height):
        for w in range(width):
            if (index[h,w] == 0):
                result_im[h, w, :] = 0


    for h in range(1, height - 1):
        for w in range(1, width - 1):
            if (index[h, w] != index[h - 1, w] or index[h, w] != index[h + 1, w] or
                    index[h, w] != index[h, w - 1] or index[h, w] != index[h, w + 1]):
                result_im[h, w, 0] = 255
                result_im[h, w, 1] = 0
                result_im[h, w, 2] = 0
    return result_im

def component_processing(index):
    index_2 = mini_object_removal(index)
    start = time.time()
    component = component_cal(index_2)
    end = time.time()
    t = end - start
    return index_2, component, t












