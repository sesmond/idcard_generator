# -*- coding: utf-8 -*- 
# @Time : 2019/11/6 4:40 下午 
# @Author : minjianxu
# @File : test.py

import  numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
import  cv2

print(np.arange(1965,1995))

print(np.arange(1,13))

# print(day_arr)

# for i in range(100):
# ?    print(i)

# for a in range(1,10,2):
#     # print(a)
#     # print(np.random.randint(12))
#     # print(np.random.choice(['10','20','30']))
#     print(str(np.random.randint(100,999)))
#

im = cv2.imread('../resource/empty.png')
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()

img_hsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
plt.imshow(img_hsv)
plt.show()

lower_blue=np.array([0,0,0]) #获取最小阈值

upper_blue=np.array([0,255,255]) #获取最大阈值

mask = cv2.inRange(img_hsv, lower_blue, upper_blue) #创建遮罩
plt.imshow(mask)
plt.show()


#
# x,y = im.size
# try:
#   p = Image.new('RGBA', im.size, (255,255,255))
#   plt.imshow(p)
#   plt.show()
#   p.paste(im, (0, 0, x, y), im)
#   p.save('new.png')
# except:
#     pass

# back_img = cv2.cvtColor(back_img,cv2.COLOR_RGB2BGR) #图像格式转换