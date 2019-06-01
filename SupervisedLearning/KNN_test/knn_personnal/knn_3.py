# -*- coding:utf8 -*-
# @TIME     :2019/5/28 21:35
# @Author   : 洪松
# @File     : knn_3.py

from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
from os import listdir
import time

'''
函数说明:将32x32的二进制图像转换为1x1024向量。
'''
def img_into_vector(filename):
    # 创建1x1024零向量
    return_vector = np.zeros((1, 1024))
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        line_str = fr.readline()
        # 每一行的前32个元素依次添加到 return_vector中
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_str[j])
    return return_vector

'''
函数说明:手写数字分类测试
'''
def hand_writing_class_test():
    # 测试集的labels
    test_labels = []
    # 返回trainingDigits目录下的文件名
    training_file_list = listdir(r'D:\书籍PDF\机器学习实战代码\kNN\3.数字识别\trainingDigits')
    # 返回文件夹下文件的个数
    m = len(training_file_list)
    # 初始化训练的mat矩阵,测试集
    training_mat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        file_name_tarin_str = training_file_list[i]
        # 获得分类的数字
        class_number_train = int(file_name_tarin_str.split('_')[0])
        # 将获得的类别添加到test_labels中
        test_labels.append(class_number_train)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        training_mat[i, :] = img_into_vector(r'D:\书籍PDF\机器学习实战代码\kNN\3.数字识别\trainingDigits\%s' % (file_name_tarin_str))
    # 构建KNN分类器
    neigh = KNN(n_neighbors=3, algorithm='auto') #algorithm参数是auto，更改algorithm参数为brute，使用暴力搜索，你会发现，运行时间变长了
    # 拟合模型, training_mat为测试矩阵,test_labels为对应的标签
    neigh.fit(training_mat, test_labels)
    test_file_list = listdir(r'D:\书籍PDF\机器学习实战代码\kNN\3.数字识别\testDigits')
    # 错误检测计数
    error_count = 0.0
    # 测试数据的数量
    m_test = len(test_file_list)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(m_test):
        # 获得文件名字
        file_name_test_str = test_file_list[i]
        # 获得分类的数字
        class_number_test = int(file_name_test_str.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vector_test = img_into_vector(r'D:\书籍PDF\机器学习实战代码\kNN\3.数字识别\testDigits\%s' % (file_name_test_str))
        # 获得预测结果
        classifier_result = neigh.predict(vector_test)
        print("分类返回结果为%d\t真实结果为%d" % (classifier_result, class_number_test))
        if (classifier_result != class_number_test):
            error_count += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (error_count, error_count / m_test * 100))

'''
kNN算法的优缺点

优点

简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
可用于数值型数据和离散型数据；
训练时间复杂度为O(n)；无数据输入假定；
对异常值不敏感。

缺点：

计算复杂性高；空间复杂性高；
样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
最大的缺点是无法给出数据的内在含义。

'''

if __name__ == '__main__':
    s = time.time()
    hand_writing_class_test()
    e = time.time()
    print(e - s)