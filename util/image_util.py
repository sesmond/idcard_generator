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
        img = cv2.imread(directory_name + "/" + filename)
        icon_all.append(img)
    print(len(icon_all))

def getIcon():
    '''
    随机获取一张图片
    :return:
    '''
    return icon_all[random.randint(0,len(icon_all))]


if __name__ == '__main__':
    initIcon()
    img = getIcon()
    cv2.imshow("name",img)
    cv2.waitKey(0)
