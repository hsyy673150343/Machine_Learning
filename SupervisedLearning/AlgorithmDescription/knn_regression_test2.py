# -*- coding:utf8 -*-
# @TIME     :2019/6/1 21:46
# @Author   : 洪松
# @File     : knn_regression_test2.py

from sklearn.neighbors import KNeighborsRegressor
from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn
X, y = make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(15,4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
axes[0].legend(['Model prediction', 'Training data/target', 'Test data/target'], loc='best')
plt.show()