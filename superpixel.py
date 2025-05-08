import cv2
import numpy as np
import math
import os
import time
import heapq
from itertools import chain
from numpy import linalg as LA

from matplotlib import pyplot as plt


def SLIC(img, size):
    ### SLIC 算法
    # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20

    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=size, ruler= 10)
    slic.iterate(10)     # 迭代次数，越大效果越好
    #mask_slic = slic.getLabelContourMask()     # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()     # 获取超像素标签

    return label_slic


if __name__== '__main__':


        image_file_name = r''

        image = plt.imread(image_file_name)
