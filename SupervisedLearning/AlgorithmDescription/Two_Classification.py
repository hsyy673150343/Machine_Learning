# -*- coding:utf8 -*-
# @TIME     :2019/6/1 19:38
# @Author   : 洪松
# @File     : Two_Classification.py
import matplotlib.pyplot as plt
from mglearn import datasets, discrete_scatter

'''生成数据集'''
X, y = datasets.make_forge()
'''数据集绘图'''
discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()
print('X.shape: {}'.format(X.shape))

