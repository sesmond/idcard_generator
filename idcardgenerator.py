# coding:utf-8
import os
import PIL.Image as PImage
from PIL import ImageFont, ImageDraw
import cv2
import numpy as np
import util.config_util as util
from util import  config
from util import  image_util
import sys
from entity.idcard import IdCard

import matplotlib.pyplot as plt


"""

“姓名”、“性别”、“民族”、“出生年月日”、“住址”、“公民身份号码”为6号黑体字，用蓝色油墨印刷;
登记项目中的姓名项用5号黑体字印刷;其他项目则用小5号黑体字印刷;

身份证上自己的姓名的字体是什么字体？

出生年月日 方正黑体简体
字符大小：姓名＋号码（11点）其他（9点）
字符间距（AV）：号码（50）
字符行距：住址（12点）

身份证号码字体 OCR-B 10 BT 文字 华文细黑

其右侧为证件名称“中华人民共和国居民身份证”，分上下两排排列，
其中上排的“中华人民共和国”为4号宋体字，
下排的“居民身份证”为2号宋体字;
“签发机关”、“有效期限”为6号加粗黑体字;
签发机关登记项采用，“xx市公安局”;
有效期限采用“xxxx.xx-xxxx.xx.xx”格式，使用5号黑体字印刷

### 生成需求：

- 各种信息自动生成，地址变长
- 对生成图，做各类仿射和透射
- 对生成图，做各类模糊和噪音、光照处理
- 选择多张背景进行贴片
- 生成仿射后的EAST训练数据
"""

if getattr(sys, 'frozen', None):
    base_dir = os.path.join(sys._MEIPASS, 'resource')
else:
    base_dir = os.path.join(os.path.dirname(__file__), 'resource')

print(base_dir)
def changeBackground(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    # lower_blue = np.array([78, 43, 46])
    # upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] + j] = img[i, j]  # 此处替换颜色，为BGR通道

    return img_back


def paste(avatar, bg, zoom_size, center):
    avatar = cv2.resize(avatar, zoom_size)
    rows, cols, channels = avatar.shape
    for i in range(rows):
        for j in range(cols):
            bg[center[0] + i, center[1] + j] = avatar[i, j]
    return bg


def generator(idCard, image_name):
    addr = idCard.addr

    # 加载空模板
    im = PImage.open(os.path.join(base_dir, 'empty.png'))
    # im = cv2.imread(os.path.join(base_dir, 'empty.png'))
    # TODO 头像 随机选

    # avatar = PImage.open(fname)  # 500x670
    avatar = card.avatar

    name_font = ImageFont.truetype(os.path.join(base_dir, 'font/hei.ttf'), 72)
    other_font = ImageFont.truetype(os.path.join(base_dir, 'font/hei.ttf'), 60)
    bdate_font = ImageFont.truetype(os.path.join(base_dir, 'font/fzhei.ttf'), 60)
    id_font = ImageFont.truetype(os.path.join(base_dir, 'font/ocrb10bt.ttf'), 72)

    draw = ImageDraw.Draw(im)
    draw.text((630, 690), idCard.name, fill=(0, 0, 0), font=name_font)
    draw.text((630, 840), idCard.sex, fill=(0, 0, 0), font=other_font)
    draw.text((1030, 840), idCard.nation, fill=(0, 0, 0), font=other_font)
    draw.text((630, 980), idCard.year, fill=(0, 0, 0), font=bdate_font)
    draw.text((950, 980), idCard.month, fill=(0, 0, 0), font=bdate_font)
    draw.text((1150, 980), idCard.day, fill=(0, 0, 0), font=bdate_font)
    # 地址
    start = 0
    loc = 1120
    while start + 11 < len(addr):
        draw.text((630, loc), addr[start:start + 11], fill=(0, 0, 0), font=other_font)
        start += 11
        loc += 100
    draw.text((630, loc), addr[start:], fill=(0, 0, 0), font=other_font)
    draw.text((950, 1475), idCard.idNo, fill=(0, 0, 0), font=id_font)
    draw.text((1050, 2750), idCard.org, fill=(0, 0, 0), font=other_font)
    draw.text((1050, 2895), idCard.validPeriod, fill=(0, 0, 0), font=other_font)

    avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
    plt.imshow(avatar)
    plt.show()
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    plt.imshow(im)
    plt.show()

    im = changeBackground(avatar, im, (500, 670), (690, 1500))

    plt.imshow(im)
    plt.show()

    im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))

    plt.imshow(im)
    plt.show()

    #TODO 贴背景图

    # (left, upper, right, lower) x1,y1,x2,y2
    front = im.crop([275, 480, 2180, 1680])
    back = im.crop([275, 1897, 2180, 3104])

    front_size = front.size
    front = front.resize((front_size[0]//config.SCALE_RATE,front_size[1]//config.SCALE_RATE))

    front.save('data/color_' + image_name + '_front.png')

    back_size = back.size
    back = back.resize((back_size[0] // config.SCALE_RATE, back_size[1] // config.SCALE_RATE))

    back.save('data/color_' + image_name + '_back.png')

    # im.convert('L').save('data/bw_'+image_name+'.png')
    print('成功', u'文件已生成到目录下', image_name)
    # showinfo(u'成功', u'文件已生成到目录下,黑白bw.png和彩色color.png')


if __name__ == '__main__':
    # TODO 赋值
    util.initArea()
    image_util.initIcon()

    for i in range(0,1):
        card = util.generateIdCard()
        card.print()
        card.avatar = image_util.getIcon()
        img_name = 'id_' + str(i).zfill(5)
        generator(card, img_name)
