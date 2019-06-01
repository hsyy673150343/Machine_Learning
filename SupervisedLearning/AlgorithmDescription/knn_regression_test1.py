# -*- coding:utf8 -*-
# @TIME     :2019/6/1 21:37
# @Author   : 洪松
# @File     : knn_regression_test1.py

from sklearn.neighbors import KNeighborsRegressor
from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split

X, y = make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print('Test set prediction:\n{}'.format(reg.predict(X_test)))
print('Test set R^2: {:.2f}'.format(reg.score(X_test,y_test)))