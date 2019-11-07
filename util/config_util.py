"""
提供config的一些读取方法
@Time : 2019/11/6 4:02 下午
@Author : minjianxu
@File : config_util.py
"""
from util import config
import numpy as np
from numpy import random

from entity.idcard import IdCard
list = []  # 地址


# 获取姓名
def getName():
    xing_len = len(config.xing)
    xing_idx = np.random.randint(0, xing_len)
    xing = config.xing[xing_idx]
    ming_len = len(config.mingzi)
    ming_idx = np.random.randint(0, ming_len)
    ming = config.mingzi[ming_idx]
    return xing + ming


def initArea():
    for i in range(0, len(config.area), 2):
        if len(config.area[i]) > 5:
            ar = [config.area[i], config.area[i + 1]]
            list.append(ar)
    print("初始化 配置参数")


def getAddress():
    index = random.randint(len(list))
    str = list[index]
    addess = [str[0], str[1] \
              + config.address3[random.randint(len(config.address3))], str[1]]
    return addess


def getIdcode(num):
    '''
        生成身份证号码
    :param num: 所在地编码
    :return:
    '''
    #TODO 年月日
    idCode = [1]*5
    # 年
    idCode[0] = str(random.randint(1965,2000))
    # 月
    idCode[1] = str(random.randint(1,13)).zfill(2)
    # 日
    idCode[2] = str(random.randint(1,29)).zfill(2)
    #三位校验码
    str3 = str(random.randint(100,999))
    pinCode = num + idCode[0] + idCode[1] + idCode[2] \
            + str3

    idCode[3] = random.choice(["男","女"])
    # 最后一位
    veri_code = [ '1', '0', 'X', '9', '8', '7', '6','5', '4', '3', '2']
    pinCode = pinCode + random.choice(veri_code)
    idCode[4] = pinCode
    return idCode

def getExpDate():
    month = str(random.randint(1,13)).zfill(2)
    day = str(random.randint(1,29)).zfill(2)
    start_year = random.randint(2006, 2019)
    year_len = random.choice([10, 20, 100])
    # 有效日期
    exp_start = str(start_year) + "." + \
                month + "." + day + "-"
    if year_len == 100:
        exp_end = "长期"
    else:
        exp_end = str(start_year + year_len) + "." + \
                  month + "." + day
    return exp_start + exp_end


def getZu():
    return config.zu[random.randint(len(config.zu))]


def generateIdCard():
    """
    随机生成IdCard
    :return:
    """
    card = IdCard()
    addr = getAddress()
    idCode = getIdcode(addr[0])
    card.name = getName()
    card.sex = idCode[3]
    #民族
    card.nation = getZu()
    card.year = idCode[0]
    card.month = idCode[1]
    card.day = idCode[2]
    card.addr = addr[1]
    # 证件号码
    card.idNo = idCode[4]
    #签发机关
    card.org = addr[2] + "公安局"
    # 有效期限
    card.validPeriod = getExpDate()
    return  card

if __name__ == '__main__':
    print(getName())
    print(getZu())
    initArea()
    print(getAddress())
    print(getExpDate())
    print(getIdcode('12345'))
