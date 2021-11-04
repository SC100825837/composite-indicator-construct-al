import pandas as pd
import numpy as np
import math
from numpy import array

# 1读取数据
df = pd.read_csv("raw_data_23.csv")

# list1 = [[5, 1.4, 6, 3, 5, 7], [9, 2, 30, 7, 5, 9], [8, 1.8, 11, 5, 7, 5], [12, 2.5, 18, 7, 5, 5]]
# df = pd.DataFrame(list1, columns=["a", "b", "c", "d", "e", "f"])
# print(df)

data = pd.DataFrame(df)

# 2数据预处理 ,去除空值的记录
data.dropna()


# 定义熵值法函数
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)
    print(k)

    lnf = [[None] * cols for i in range(rows)]

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
    print(E)
    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
    w = pd.DataFrame(w)
    return w


if __name__ == '__main__':
    # 计算df各字段的权重
    w = cal_weight(data).round(4)  # 调用cal_weight
    w.index = data.columns
    w.columns = ['weight']
    print(w)  # 输出权重
    print('熵权法计算权重运行完成!')
