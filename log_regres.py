
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/10/15 14:22
# @Author  :  duyuheng
# @Site     : 
# @File     : log_regres.py
# @Software  : PyCharm




def load_data_set():
    data_mat = []
    label_mat = []
    fr =open('testSet.txt')
    for line in fr.readline():
        line_arr = line.strip().split()
        data_mat.append([1.o, float(line_arr[0]),float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m,n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat-h)
        weights = weights + alpha* data_matrix.transpose() * error
    return weights



