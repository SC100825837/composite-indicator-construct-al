from SALib.sample import saltelli
from SALib.analyze import sobol
import json

import sys

import numpy as np

str1 = "[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]"
str2 = "8"
str3 = "[[8.0, 994.0], [4.0, 156.6], [4.8, 200.2], [15.4, 80.8], [2.73, 3.12], [3.46, 4.39], [7.1, 12.0], [7.7, 27.4]]"


def robustness(weight, num, arrbounds):
    # print(type(weight))  # list
    # print(type(num)) # int
    # print(type(arrbounds)) # list
    # 定义模型输入
    problem = {  # problem 为一个字典类型的数据
        # 输入的变量个数
        'num_vars': num,
        'bounds': arrbounds
    }

    # 生成样本数据
    X = saltelli.sample(problem, 1024)

    arr = []
    for x in X:
        sum = 0
        for i in range(0, len(x)):
            sum += x[i] * weight[i]
        arr.append(sum)
    Y = np.array(arr)

    Si = sobol.analyze(problem, Y)
    return Si['S1']


if __name__ == '__main__':

    # print(json.loads(sys.argv[1]))

    weight = json.loads(sys.argv[1])
    count = int(sys.argv[2])
    bounds = json.loads(sys.argv[3])
    # print(weight)
    # print(count)
    # print(bounds)


    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    # weight = json.loads(str1)
    # count = int(str2)
    # bounds = json.loads(str3)

    result = robustness(weight, count, bounds)

    print(result)
