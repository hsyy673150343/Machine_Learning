# -*- coding:utf8 -*-
# @TIME     :2019/6/1 20:10
# @Author   : 洪松
# @File     : Two_Useful_datasets.py

from sklearn.datasets import load_breast_cancer, load_boston
import numpy as np
import mglearn
'''威斯康辛州乳腺癌数据集'''
cancer = load_breast_cancer()
print('cancer.key():\n{}:'.format(cancer.keys()))
print('Shape of cancer data:{}'.format(cancer.data.shape))
print('Target_names：{}'.format(cancer.target_names))
print('Target：{}'.format(cancer.target))
'''np.bincount()：统计次数'''
print('Sample counts per class:\n{}'.format({n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

'''波士顿房价数据集'''
boston = load_boston()
print('load_boston.keys():\n{}'.format(boston.keys()))
print('Data shape: {}'.format(boston.data.shape))

X,y = mglearn.datasets.load_extended_boston()
print('X.shape:{}'.format(X.shape))