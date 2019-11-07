#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :图片处理工具类
@File    :   image_util.py
@Author  : minjianxu
@Time    : 2019/11/7 5:25 下午
@Version : 1.0
'''

import  cv2
import  os
import  random

icon_all = []

def initIcon():
    '''
    初始化头像
    :return:
    '''
    global  icon_all
    directory_name = "resource/icon"
    for filename in os.listdir(r"./" + directory_name):
        if(filename.endswith('g')):
            img = cv2.imread(directory_name + "/" + filename)
        # print(filename)
        # plt.imshow(img)
        # plt.show()
            icon_all.append(img)
    print(len(icon_all))

def getIcon():
    '''
    随机获取一张图片
    :return:
    '''
    return icon_all[random.randint(0,len(icon_all))]

import matplotlib.pyplot as plt
if __name__ == '__main__':
    # img = getIcon()
    # cv2.imshow("name",img)
    # cv2.waitKey(0)
    # for icon icon_all
    initIcon()
    for i in range(0, 36):
        # cv2.imshow('name',icon_all[i])
        # cv2.waitKey(0)
        plt.imshow(icon_all[i])
        plt.show()
        cv2.imwrite("resource/icon/" + str(i).zfill(5) + '.png', icon_all[i])
    #
    #     print(i)
