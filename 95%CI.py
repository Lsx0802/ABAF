## 记于2021-10-10
## 近似服从正态分布的情况下，如何计算置信区间
## 置信水平 90%--z=1.64   95%--z=1.96   99%--z=2.58
## 计算公式：(mean-z*std, mean+z*std)

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
## 93.89 ± 1.87（均值+均差）83.31 ± 2.43
mean = 73.51
std = 3.16
prob = stats.norm.pdf(0,mean,std) #在0处概率密度值
pre = stats.norm.cdf(0,mean,std)  #预测小于0的概率
interval = stats.norm.interval(0.95,mean,std)  # 95%置信水平的区间
print(interval) #输出结果

from scipy import stats
import numpy as np

arr = [0] * 57 + [1] * 43
arr = np.array(arr)
mean = arr.mean()
std = arr.std(ddof=1) / np.sqrt(len(arr))  # 注意ddof
confidence_rate = 0.95
interval = stats.norm.interval(confidence_rate, mean, std)  # 置信水平的区间
print('{}%的置信区间是{}'.format(round(confidence_rate * 100, 3), interval))


