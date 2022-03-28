import numpy as np
import scipy, math
from scipy.stats import f
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# additional packages
# from statsmodels.stats.diagnostic import lillifors
# 多重比较
# from statsmodels.sandbox.stats.multicomp import multipletests
# 用于排列组合
import itertools

a = 0.05


# LSD(least significant difference)最小显著差异list_groups组的个数  n_group每组测试数据数
def LSD(list_groups, list_total, n_group, mean1, mean2, std1, std2):
    distance = abs(mean1 - mean2)
    print("distance:", distance)
    # t检验的自由度
    df = list_total - 1 * list_groups  #总样本数  有多少个组
    # mse = MSE(list_groups, list_total)  #样本均方误差
    # print("MSE:", mse)
    mse = (std1+std2)/2.0
    print('mse:', mse)
    t_value = stats.t(df).isf(a / 2.0)
    print("t value:", t_value)
    lsd = t_value * math.sqrt(mse * (1.0 / n_group + 1.0 / n_group))
    print("LSD:", lsd)
    if distance < lsd:
        print("no significant difference between:", mean1, mean2)
    else:
        print("there is significant difference between:", mean1, mean2)

LSD(list_groups=3, list_total=3*10, n_group=10, mean1=77.00, mean2=76.50, std1=1.87, std2=3.74)

