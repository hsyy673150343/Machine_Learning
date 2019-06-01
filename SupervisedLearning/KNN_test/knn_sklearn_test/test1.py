# -*- coding:utf8 -*-
# @TIME     :2019/5/30 17:21
# @Author   : 洪松
# @File     : Two_Classification.py

from sklearn.datasets import load_iris
'''鸢尾花数据集'''

iris_dataset = load_iris()
print('Key of iris_dataset: \n{}'.format(iris_dataset.keys()))
print(iris_dataset['DESCR'])
print('Target_names: {}'.format(iris_dataset['target_names']))
print('Feature_names: {}'.format(iris_dataset['feature_names']))
print('Data: {}'.format(type(iris_dataset['data'])))
print('Type 0f data:\n {}'.format(iris_dataset['data']))
print('Shape 0f data:\n {}'.format(iris_dataset['data'].shape))
print('Type 0f target: {}'.format(type(iris_dataset['target'])))
print('Shape 0f target: {}'.format(iris_dataset['target'].shape))
print('Target:\n {}'.format(iris_dataset['target']))
