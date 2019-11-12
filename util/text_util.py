#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   : 文本处理工具类
@File    :   text_util.py    
@Author  : minjianxu
@Time    : 2019/11/11 11:28 上午
@Version : 1.0 
'''


def caculate_text_shape(text, font):
    '''
    计算文本框大小 方便计算坐标
    :param text:
    :param font:
    :return:
    '''
    # 获得文字的offset位置
    offsetx, offsety = font.getoffset(text)
    # 获得文件的大小,font.getsize计算的比较准
    width, height = font.getsize(text)

    width = width  # - offsetx
    height = height  # - offsety

    return width, height


def calculate_text_box(x, y, text, font):
    """
    计算文本框坐标
    :param x: 画图左上角坐标x
    :param y: 画图左上角坐标y
    :param text: 文本
    :param font: 字体
    :return: 返回文本框坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    points = [(0,0), (original_width,0), (original_width,original_height), (0,original_height)]
        左上，右上，右下，左下
    """
    w, h = caculate_text_shape(text, font)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y
    x3 = x + w
    y3 = y + h
    x4 = x
    y4 = y + h
    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def calculate_scale_box(boxes,scale):
    '''
    缩放图片时，box重新计算
    :param boxes:
    :param scale: 缩放比例
    :return:
    '''
    #TODO 如何取值 是否要加减像素 上取整还是下取整
    for box in boxes:
        for pts in box:
            pts[0] =  int(pts[0]/scale)
            pts[1] = int(pts[1]/scale)
    # boxes = boxes /scale
    return  boxes

def generate_box_by_two(x1, y1, x2, y2):
    '''
    两点坐标生成四点坐标
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    '''
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def move_box_coordinate(add_x, add_y, boxes):
    '''
    小框坐标截取
    :param add_x: 左上X
    :param add_y: 左上Y
    :param boxes:
    :return:
    '''
    # box 全部加+，试试批量操作
    if (boxes != None and len(boxes) > 0):
        new_boxes = []
        for box in boxes:
            new_box = []
            for point in box:
                new_point = [0] * 2
                new_point[0] = point[0] - add_x
                new_point[1] = point[1] - add_y
                new_box.append(new_point)
            new_boxes.append(new_box)
        return new_boxes
    else:
        return None
