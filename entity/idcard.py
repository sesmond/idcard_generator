# -*- coding: utf-8 -*- 
# @Time : 2019/11/7 11:37 上午 
# @Author : minjianxu
# @File : idcard.py


class IdCard:
    def __init__(self):
        self._name = ''
        self._sex = ''
        self._nation = ''
        self._year = ''
        self._month = ''
        # 日期
        self._day = ''
        # 住址
        self._addr = ''
        # 身份证号码
        self._idNo = ''
        # 签发机关
        self._org = ''
        # 有效期
        self._validPeriod = ''
        # 头像
        self._avatar = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def sex(self):
        return self._sex

    @sex.setter
    def sex(self, value):
        self._sex = value

    @property
    def nation(self):
        return self._nation

    @nation.setter
    def nation(self, value):
        self._nation = value

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        self._year = value

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, value):
        self._month = value

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = value

    @property
    def addr(self):
        return self._addr

    @addr.setter
    def addr(self, value):
        self._addr = value

    @property
    def idNo(self):
        return self._idNo

    @idNo.setter
    def idNo(self, value):
        # 身份证号码
        self._idNo = value

    @property
    def org(self):
        return self._org

    @org.setter
    def org(self, value):
        self._org = value

    @property
    def validPeriod(self):
        return self._validPeriod

    @validPeriod.setter
    def validPeriod(self, value):
        self._validPeriod = value

    @property
    def avatar(self):
        return self._avatar

    @avatar.setter
    def avatar(self, value):
        self._avatar = value

    def print(self):
        print('name:',self._name,
              'sex:',self._sex,
              'nation:', self._nation,
              'idNo:',self._idNo,
              'addr:',self._addr,
              'org:',self._org,
              'year:',self._year,
              'month:',self._month,
              'day:',self._day,
              )


if __name__ == '__main__':
    card = IdCard()
    card.name = '1293'
    print(card.name)
