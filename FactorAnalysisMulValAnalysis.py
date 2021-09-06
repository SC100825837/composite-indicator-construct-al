import numpy.linalg as nlg
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import math
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import array
import sys


# 加载数据
def load_data(data_str):
    # print(data)
    row_list = data_str.replace(' ', '').replace('[', '').split("],")
    row_list[len(row_list) - 1] = row_list[len(row_list) - 1][:-2]
    # [[0] * cols for i in range(rows)]

    origin_data_arr = [[0] * 8 for i in range(len(row_list))]
    # print(origin_data_arr)
    for i in range(len(row_list)):
        row = row_list[i].split(",")
        for j in range(len(row)):
            origin_data_arr[i][j] = float(row[j])

    origin_data_df = DataFrame(origin_data_arr)
    origin_data_df.rename(columns={0: 'patents', 1: 'royalties', 2: 'internet', 3: 'exports',
                                   4: 'telephones', 5: 'electricity', 6: 'schooling', 7: 'university'}, inplace=True)
    # print(origin_data_df)
    return origin_data_df


# 相关系数矩阵C
def correlation_coefficient_matrix(data):
    corr = data.corr()
    # print(corr)
    return corr


# 相关系数矩阵C
def correlation_coefficient_matrix_arr(data):
    corr = data.corr()
    return corr.values.tolist()


# principal主成分分析，varimax因子旋转
def init_fa(data):
    fa = FactorAnalyzer(n_factors=4, method='principal', rotation="varimax")
    fa.fit(data)
    return fa


# 成分旋转矩阵（旋转因子负荷矩阵），可以看出特征的归属因子
def rotated_factor_loadings_matrix_arr(fa):
    loadings = fa.loadings_
    Row = 8  # 二维数组中一位列表的个数
    Col = len(loadings[0])  # 每个列表的个数
    arr = [[] * Row for i in range(Col)]  # 给定矩阵长、宽 创建一个空的二维数组
    for i in range(Row):
        for j in range(Col):
            arr[j].append(loadings[i][j])
    # print(loadings)
    # print(arr)
    return arr


def rotated_factor_loadings_matrix(fa):
    return fa.loadings_


# 共同性
def get_communalities(fa):
    communality = fa.get_communalities()
    # print(communality)
    return communality


if __name__ == '__main__':
    data = load_data(sys.argv[1])
    # print(data)
    corr_arr = correlation_coefficient_matrix_arr(data)
    print(corr_arr)
    fa = init_fa(data)
    loadings = rotated_factor_loadings_matrix(fa)
    loadings_arr = rotated_factor_loadings_matrix_arr(fa)
    # print(loadings)
    # communality = get_communalities(fa)
    # print(weight)
    print(loadings_arr)

# ml最大似然，varimax最大方差归一化旋转
# fa = FactorAnalyzer(n_factors=4, method='ml', rotation="varimax")
# fa.fit(data)
#
# # 共同性
# communality = fa.get_communalities()
# # 成分矩阵，可以看出特征的归属因子
# loadings = fa.loadings_
# plt.figure()
# ax = sns.heatmap(loadings, annot=True, cmap="BuPu")
# plt.title('Factor Analysis')
# # 贡献率
# factor_variance = fa.get_factor_variance()
# # 因子得分
# fa_score = fa.transform(data)
# plt.show()
# factor_variance[0]旋转之后的特征值，factor_variance[1]解释度，factor_variance[2]积累的
# print(factor_variance)
