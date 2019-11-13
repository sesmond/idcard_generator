from PIL import Image
import random
from util import image_util

import matplotlib.pyplot as plt


def random_rotate_paste(origin_img, bg_img):
    #
    img_a = origin_img.convert('RGBA')
    img_r = img_a.rotate(random.randint(-90, 90), expand=1)
    # 如果图片大于背景图片那就贴不全了，需要放大背景
    w0, h0 = img_r.size
    w1, h1 = bg_img.size
    print(img_r.size,bg_img.resize)
    if w1 < w0 or h1 < h0:
        # resize 背景大小
        w1 = 2 * w0
        h1 = 2 * h0
        bg_img = bg_img.resize((w1, h1))
    add_x = random.randint(0, w1 - w0)
    add_y = random.randint(0, h1 - h0)

    print(img_r.size, bg_img.resize,add_x,add_y)
    bg_img.paste(img_r, (add_x, add_y), img_r)
    return bg_img


def show(img_p):
    plt.imshow(img_p)
    plt.show()


if __name__ == '__main__':
    img = Image.open("util/3.jpg")
    bg_list = image_util.get_all_bg_images()
    for i in range(10):
        bg, w, h = image_util.create_backgroud_image(bg_list)
        print(w, h)
        bg = random_rotate_paste(img, bg)
        show(bg)
