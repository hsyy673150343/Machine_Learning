# -*- coding:utf8 -*-
# @TIME     :2019/6/1 20:37
# @Author   : 洪松
# @File     : knn_classification_test1.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_forge
import matplotlib.pyplot as plt
from mglearn import plots, discrete_scatter

'''X--数据 y--标签'''
X, y = make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print('Test set prediction: {}'.format(clf.predict(X_test)))
print('Test set accuracy: {:.2}'.format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title('{} neighbors(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
axes[0].legend(loc=3)
plt.show()