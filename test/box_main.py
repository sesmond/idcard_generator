#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :TODO
@File    :   box_main.py    
@Author  : minjianxu
@Time    : 2019/11/11 12:52 下午
@Version : 1.0 
'''

from  util import  image_util
import numpy as np
import  math

# boxes = [[[0,0],[1,0],[1,1],[0,1]]]
boxes = [[[100,100]]]
print(boxes)
# for box in boxes:
#     new_box = np.array(box,dtype=float)
#     print(new_box)
#     print(np.array([new_box]))
    # box[0]+=10
    # box[1]+=20
# print(boxes)
# src=[100,100]
# center=[50,50]
# image_util.getPointAffinedPo(src,center,90)
center=(50,50)
angle = 90

angle1 = angle * math.pi / 180
angle2 = math.radians(-angle)
print(angle1,angle2)

image_util.get_rotate_box(boxes,center,angle)
# # theta = math.radians(90)
# theta = angle
# print("theta",theta)
# for box in boxes:
#     for pts  in box:
#         print('前：',pts)
#         image_util.ge
#         pts = image_util._rotate_one_point(pts,center,theta)
#         print('后：', pts)

# boxes[:, 0]