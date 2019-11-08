#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :图片处理工具类
@File    :   image_util.py
@Author  : minjianxu
@Time    : 2019/11/7 5:25 下午
@Version : 1.0
'''

import cv2
import os, math
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

DEBUG = False
ROOT = "resource"  # 定义运行时候的数据目录，原因是imgen.sh在根部运行
DATA_DIR = "data"
MAX_LENGTH = 20  # 可能的最大长度（字符数）
MIN_LENGTH = 1  # 可能的最小长度（字符数）
MAX_FONT_SIZE = 28  # 最大的字体
MIN_FONT_SIZE = 18  # 最小的字体号
MAX_LINE_HEIGHT = 100  # 最大的高度（像素）
MIN_LINE_HEIGHT = MIN_FONT_SIZE + 12  # 最小的高度（像素）

# 颜色的算法是，产生一个基准，然后RGB上下浮动FONT_COLOR_NOISE
MAX_FONT_COLOR = 100  # 最大的可能颜色
FONT_COLOR_NOISE = 10  # 最大的可能颜色
ONE_CHARACTOR_WIDTH = 1024  # 一个字的宽度
ROTATE_ANGLE = 90  # 随机旋转角度
GAUSS_RADIUS_MIN = 0.5  # 高斯模糊的radius最小值
GAUSS_RADIUS_MAX = 0.8  # 高斯模糊的radius最大值

# 之前的设置，太大，我决定改改
# MAX_BACKGROUND_WIDTH = 1600
# MIN_BACKGROUND_WIDTH = 800
# MAX_BACKGROUND_HEIGHT = 2500
# MIN_BACKGROUND_HEIGHT = 1000
MAX_BACKGROUND_WIDTH = 2000
MIN_BACKGROUND_WIDTH = 1000
MAX_BACKGROUND_HEIGHT = 2000
MIN_BACKGROUND_HEIGHT = 1000

MAX_SPECIAL_NUM = 5  # 特殊字符的个数

MAX_BLANK_NUM = 5  # 字之间随机的空格数量
MIN_BLANK_WIDTH = 50  # 最小的句子间的随机距离
MAX_BLANK_WIDTH = 100  # 最长的句子间距离

INTERFER_LINE_NUM = 10
INTERFER_POINT_NUM = 2000
INTERFER_LINE_WIGHT = 2
INTERFER_WORD_LINE_NUM = 4
INTERFER_WORD_POINT_NUM = 20
INTERFER_WORD_LINE_WIGHT = 1

# 各种可能性的概率
POSSIBILITY_BLANK = 0.8  # 有空格的概率
POSSIBILITY_ROTOATE = 0.4  # 文字的旋转
POSSIBILITY_INTEFER = 0.2  # 需要被干扰的图片，包括干扰线和点
POSSIBILITY_WORD_INTEFER = 0.1  # 需要被干扰的图片，包括干扰线和点
POSSIBILITY_AFFINE = 0.3  # 需要被做仿射的文字
POSSIBILITY_PURE_NUM = 0.2  # 需要产生的纯数字
POSSIBILITY_PURE_ENG = 0.1  # 需要产生的英语
POSSIBILITY_DATE = 0.1  # 需要产生的纯日期
POSSIBILITY_SINGLE = 0.01  # 单字的比例
POSSIBILITY_SPECIAL = 0.2  # 特殊字符

# 仿射的倾斜的错位长度  |/_/, 这个是上边或者下边右移的长度
AFFINE_OFFSET = 12


def get_all_icons():
    directory_name = "resource/icon"
    icon_list = []
    for filename in os.listdir(r"./" + directory_name):
        if (filename.endswith('g')):
            img = cv2.imread(directory_name + "/" + filename)
            icon_list.append(img)
    return icon_list


def get_all_bg_images():
    bground_path = os.path.join(ROOT, 'background/')

    bg_list = []
    for img_name in os.listdir(bground_path):
        image = Image.open(bground_path + img_name)
        if image.mode == "L":
            # logger.error("图像[%s]是灰度的，转RGB",img_name)
            image = image.convert("RGB")
        bg_list.append(image)
    return bg_list


def get_random_icon(icon_list):
    '''
    随机获取一张图片
    :return:
    '''
    img = random.choice(icon_list)
    return img.copy()


def random_rotate_paste(origin_img, bg_img):
    # plt.imshow(bg_img)
    # plt.show()
    #
    img_a = origin_img.convert('RGBA')
    img_r = img_a.rotate(random.randint(-90, 90), expand=1)
    # 如果图片大于背景图片那就贴不全了，需要放大背景
    w0, h0 = img_r.size
    w1, h1 = bg_img.size
    print(img_r.size, bg_img.resize)
    if w1 < w0 or h1 < h0:
        # resize 背景大小
        w1 = 2 * w0
        h1 = 2 * h0
        bg_img = bg_img.resize((w1, h1))
    add_x = random.randint(0, w1 - w0)
    add_y = random.randint(0, h1 - h0)

    print(img_r.size, bg_img.resize, add_x, add_y)
    bg_img.paste(img_r, (add_x, add_y), img_r)
    return bg_img


# 旋转函数
def random_rotate(img, points):
    ''' ______________
        |  /        /|
        | /        / |
        |/________/__|
        旋转可能有两种情况，一种是矩形，一种是平行四边形，
        但是传入的points，就是4个顶点，
    '''
    # TODO

    if not _random_accept(POSSIBILITY_ROTOATE): return img, points  # 不旋转

    w, h = img.size

    center = (w // 2, h // 2)

    if DEBUG: print("需要旋转")
    degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
    if DEBUG: print("旋转度数:%f" % degree)
    return img.rotate(degree, center=center, expand=1), _rotate_points(points, center, degree)


def _rotate_points(points, center, degree):
    theta = math.radians(-degree)

    original_min_x, original_min_y = np.array(points).max(axis=0)

    rotated_points = [_rotate_one_point(xy, center, theta) for xy in points]

    rotated_min_x, rotated_min_y = np.array(rotated_points).max(axis=0)

    x_offset = abs(rotated_min_x - original_min_x)
    y_offset = abs(rotated_min_y - original_min_y)

    rotated_points = [(xy[0] + x_offset, xy[1] + y_offset) for xy in rotated_points]

    return rotated_points


def _rotate_one_point(xy, center, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    cord = (
        # (xy[0] - center[0]) * cos_theta - (xy[1]-center[1]) * sin_theta + xy[0],
        # (xy[0] - center[0]) * sin_theta + (xy[1]-center[1]) * cos_theta + xy[1]
        (xy[0] - center[0]) * cos_theta - (xy[1] - center[1]) * sin_theta + center[0],
        (xy[0] - center[0]) * sin_theta + (xy[1] - center[1]) * cos_theta + center[1]

    )
    # print("旋转后的坐标：")
    # print(cord)
    return cord


# 随机仿射一下，也就是歪倒一下
# 不能随便搞，我现在是让图按照平行方向歪一下，高度不变，高度啊，大小啊，靠别的控制，否则，太乱了
def random_affine(img):
    # TODO 仿射怎么搞
    HEIGHT_PIX = 10
    WIDTH_PIX = 50

    # 太短的不考虑了做变换了
    # print(img.size)
    original_width = img.size[0]
    original_height = img.size[1]
    points = [(0, 0), (original_width, 0), (original_width, original_height), (0, original_height)]

    if original_width < WIDTH_PIX: return img, points
    # print("!!!!!!!!!!")
    if not _random_accept(POSSIBILITY_AFFINE): return img, points

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)

    is_top_fix = random.choice([True, False])

    bottom_offset = random.randint(0, AFFINE_OFFSET)  # bottom_offset 是 上边或者下边 要位移的长度

    height = img.shape[0]

    # 这里，我们设置投影变换的3个点的原则是，使用    左上(0,0)     右上(WIDTH_PIX,0)    左下(0,HEIGHT_PIX)
    # 所以，他的投影变化，要和整个的四边形做根据三角形相似做换算
    # .
    # |\
    # | \
    # |__\  <------投影变化点,  做三角形相似计算，offset_ten_pixs / bottom_offset =  HEIGHT_PIX / height
    # |   \                   所以： offset_ten_pixs = (bottom_offset * HEIGHT_PIX) / height
    # |____\ <-----bottom_offset
    offset_ten_pixs = int(HEIGHT_PIX * bottom_offset / height)  # 对应10个像素的高度，应该调整的横向offset像素
    width = int(original_width + bottom_offset)  #

    pts1 = np.float32([[0, 0], [WIDTH_PIX, 0], [0, HEIGHT_PIX]])  # 这就写死了，当做映射的3个点：左上角，左下角，右上角

    # \---------\
    # \         \
    #  \_________\
    if is_top_fix:  # 上边固定，意味着下边往右
        # print("上边左移")
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(0, 0),
                  (original_width, 0),
                  (width, original_height),
                  (bottom_offset, original_height)]
    #  /---------/
    # /         /
    # /_________/
    else:  # 下边固定，意味着上边往右
        # 得先把图往右错位，然后
        # 先右移
        # print("上边右移")
        H = np.float32([[1, 0, bottom_offset], [0, 1, 0]])  #
        img = cv2.warpAffine(img, H, (width, height))
        # 然后固定上部，移动左下角
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [-offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(bottom_offset, 0),
                  (original_width + bottom_offset, 0),
                  (width, original_height),
                  (0, original_height)]

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    return img, points


# 生成一张背景图，大小随机
def create_backgroud_image(bground_list):
    # 从背景目录中随机选一张"纸"的背景
    img = random.choice(bground_list)
    w, h = img.size
    img_new = img.crop((0, 0, w, h))
    return random_image_size(img_new)


# 随机裁剪图片的各个部分
def random_image_size(image):
    # 产生随机的大小
    height = random.randint(MIN_BACKGROUND_HEIGHT, MAX_BACKGROUND_HEIGHT)
    width = random.randint(MIN_BACKGROUND_WIDTH, MAX_BACKGROUND_WIDTH)
    # TODO 背景过大做一下随机切割
    # 高度和宽度随机后，还要随机产生起始点x,y，但是要考虑切出来不能超过之前纸张的大小，所以做以下处理：
    size = image.size
    x_scope = size[0] - width
    y_scope = size[1] - height

    if x_scope < 0 or y_scope < 0:
        image.resize((width, height))
    else:
        x = random.randint(0, x_scope)
        y = random.randint(0, y_scope)
        image = image.crop((x, y, x + width, y + height))
    # logger.debug("剪裁图像:x=%d,y=%d,w=%d,h=%d",x,y,width,height)
    # image.resize((width, height))
    return image, width, height


# # 画干扰点
def randome_intefer_point(img, possible, num):
    if not _random_accept(possible): return

    w, h = img.size
    draw = ImageDraw.Draw(img)

    point_num = random.randint(0, num)
    for i in range(point_num):
        x, y = _get_random_point(w, h)
        draw.point([x, y], _get_random_color())
    del draw


def generate_all(bground_list, image_name, label_name):
    # 先创建一张图，宽度和高度都是随机的
    image, w, h = create_backgroud_image(bground_list)

    # TODO 背景处理完再贴图还是先贴图再统一处理
    # 在整张图上产生干扰点和线
    randome_intefer_line(image, POSSIBILITY_INTEFER, INTERFER_LINE_NUM, INTERFER_LINE_WIGHT)
    randome_intefer_point(image, POSSIBILITY_INTEFER, INTERFER_POINT_NUM)

    image.save(image_name)


# 随机接受概率
def _random_accept(accept_possibility):
    return np.random.choice([True, False], p=[accept_possibility, 1 - accept_possibility])


# 画干扰线
def randome_intefer_line(img, possible, line_num, weight):
    if not _random_accept(possible): return

    w, h = img.size
    draw = ImageDraw.Draw(img)
    line_num = random.randint(0, line_num)

    for i in range(line_num):
        x1, y1 = _get_random_point(w, h)
        x2, y2 = _get_random_point(w, h)
        _weight = random.randint(0, weight)
        draw.line([x1, y1, x2, y2], _get_random_color(), _weight)

    del draw


# 产生随机颜色
def _get_random_color():
    base_color = random.randint(0, MAX_FONT_COLOR)
    noise_r = random.randint(0, FONT_COLOR_NOISE)
    noise_g = random.randint(0, FONT_COLOR_NOISE)
    noise_b = random.randint(0, FONT_COLOR_NOISE)

    noise = np.array([noise_r, noise_g, noise_b])
    font_color = (np.array(base_color) + noise).tolist()

    return tuple(font_color)


def _get_random_point(x_scope, y_scope):
    x1 = random.randint(0, x_scope)
    y1 = random.randint(0, y_scope)
    return x1, y1


import matplotlib.pyplot as plt


# plt.imshow(icon_all[i])
# plt.show()
# cv2.imwrite("resource/icon/" + str(i).zfill(5) + '.png', icon_all[i])


def add_cachet_img(cachet_img, angle, img_im, adress_x, adress_y, random_num):
    cachet_img_a = cachet_img.convert('RGBA')

    cachet_img_r = cachet_img_a.rotate(random.randrange((-1 * angle), angle))

    cachet_img_f = Image.new('RGBA', cachet_img_r.size, (255,) * 4)

    cachet_img_new = Image.composite(cachet_img_r, cachet_img_f, cachet_img_r)

    img_region = cachet_img_new.crop((0, 0, cachet_img.size[0], cachet_img.size[1]))

    img_im.paste(img_region,
                 (random.randint(adress_x, adress_x + random_num), random.randint(adress_y, adress_y + random_num)))


if __name__ == '__main__':
    # initIcon()
    # img = getIcon()

    # 生成的图片存放目录
    data_images_dir = "data/images"

    # 生成的图片对应的标签的存放目录，注意这个是大框，后续还会生成小框，即anchor，参见split_label.py
    data_labels_dir = "data/labels"

    if not os.path.exists(data_images_dir): os.makedirs(data_images_dir)
    if not os.path.exists(data_labels_dir): os.makedirs(data_labels_dir)

    for num in range(0, 100):
        image_name_1 = os.path.join(data_images_dir, str(num) + ".png")
        label_name_1 = os.path.join(data_labels_dir, str(num) + ".txt")
        print("生成：", image_name_1)
        # generate_all(,image_name,label_name)
    # logger.info("已产生[%s]",image_name)
