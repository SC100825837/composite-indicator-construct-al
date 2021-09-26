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
def load_data(data_str, column_count):
    # data = pd.read_csv('E:/Python/workspace/composite-indicator-construct-py/raw_data_23.csv')
    # print(data)
    # print('*************' + data_str)
    row_list = data_str.replace(' ', '').replace('[', '').replace(']]', '').split("],")
    # print(row_list)

    origin_data_arr = [[0] * column_count for i in range(len(row_list))]

    for i in range(len(row_list)):
        row = row_list[i].split(",")
        for j in range(len(row)):
            origin_data_arr[i][j] = float(row[j])

    origin_data_df = DataFrame(origin_data_arr)
    # origin_data_df.rename(columns={0: 'patents', 1: 'royalties', 2: 'internet', 3: 'exports',
    #                                4: 'telephones', 5: 'electricity', 6: 'schooling', 7: 'university'}, inplace=True)
    # print(origin_data_df)
    return origin_data_df


# 相关系数矩阵C
def correlation_coefficient_matrix(data):
    corr = data.corr()
    # print(corr)
    return corr


# principal主成分分析，varimax因子旋转
def init_fa(data):
    fa = FactorAnalyzer(n_factors=4, method='principal', rotation="varimax")
    fa.fit(data)
    return fa


# 成分旋转矩阵（旋转因子负荷矩阵），可以看出特征的归属因子
def rotated_factor_loadings_matrix_arr(fa, colCounts):
    loadings = fa.loadings_
    Row = colCounts  # 二维数组中一位列表的个数
    Col = len(loadings[0])  # 每个列表的个数
    arr = [[] * Row for i in range(Col)]  # 给定矩阵长、宽 创建一个空的二维数组
    for i in range(Row):
        for j in range(Col):
            arr[j].append(round(loadings[i][j], 6))
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


def get_figure(loadings, fa, data):
    plt.figure()
    ax = sns.heatmap(loadings, annot=True, cmap="BuPu")
    plt.title('Factor Analysis')
    # 贡献率
    factor_variance = fa.get_factor_variance()
    # 因子得分
    fa_score = fa.transform(data)
    # plt.show()


# 计算特征值和特征向量
# 利用变量名和特征值建立一个数据框
def get_eigenvalues_eigenvectors(corr, data, colCounts):
    eig_value, eig_vector = nlg.eig(corr)
    eig = pd.DataFrame()
    # 列名
    eig['names'] = data.columns
    # 特征值eig
    eig['特征值'] = eig_value
    # 积累方差
    arr1 = []
    # 方差百分比
    arr2 = []
    for i in range(colCounts):
        # 保留6位小数
        arr1.append(round(float(eig['特征值'][i:i + 1].sum() / eig['特征值'].sum() * 100), 6))
        arr2.append(round(float(eig['特征值'][:i + 1].sum() / eig['特征值'].sum() * 100), 6))

    eig['方差百分比'] = arr1
    eig['积累方差'] = arr2

    arr3 = []
    for i in range(len(eig_value)):
        arr3.append(round(float(eig_value[i]), 6))

    # print('arr3', arr3)
    # print('arr1', arr1)
    # print('arr2', arr2)
    arr4 = [
        arr3,
        arr1,
        arr2
    ]
    # print(arr4)
    return arr4


# factor_variance[0]旋转之后的特征值，factor_variance[1]解释度，factor_variance[2]积累的
def get_rotated_eigenvalues(fa):
    factor_variance = fa.get_factor_variance()
    # print(factor_variance)
    # print(factor_variance[0])
    return factor_variance


# Expl./Tot是解释方差除以总方差四个因素
def get_Expl_Tot(factor_variance):
    arr = ['0', '0', '0', '0']
    sum1 = round(factor_variance[0][0] + factor_variance[0][1] + factor_variance[0][2] + factor_variance[0][3], 2)
    for i in range(0, 4):
        arr[i] = round(round(factor_variance[0][i], 2) / sum1, 2)
    # print(arr)
    return arr


# 定义熵值法函数
def cal_weight(x):
    """熵值法计算变量的权重"""
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k'
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    # 矩阵计算--
    # 信息熵
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        w[j] = round(d[j] / sum(d), 6)
        # 计算各样本的综合得分,用最原始的数据
    # w = pd.DataFrame(w)
    # print(w)
    return w


if __name__ == '__main__':
    data = load_data(sys.argv[1], int(sys.argv[2]))
    # print(data)
    corr = correlation_coefficient_matrix(data)
    fa = init_fa(data)
    loadings = rotated_factor_loadings_matrix(fa)
    loadings_arr = rotated_factor_loadings_matrix_arr(fa, int(sys.argv[2]))
    # print('loadings_arr', loadings_arr)
    communality = get_communalities(fa)
    # print('corr', corr)
    eig = get_eigenvalues_eigenvectors(corr, data, int(sys.argv[2]))
    # print('eig', eig)
    factor_variance = get_rotated_eigenvalues(fa)
    expl_tot = get_Expl_Tot(factor_variance)
    weight = cal_weight(pd.DataFrame(list(zip(*loadings))))
    # print('weight', weight)
    result = [
        loadings_arr,
        eig,
        weight
    ]
    print(result)

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
