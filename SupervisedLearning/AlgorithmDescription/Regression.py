# -*- coding:utf8 -*-
# @TIME     :2019/6/1 20:01
# @Author   : 洪松
# @File     : Regression.py

import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()