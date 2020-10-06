# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from scipy import stats
from scipy import optimize as opt
from scipy.stats import beta, uniform  # ベータ分布と一様分布
import matplotlib.pyplot as plt
# %matplotlib inline

plt.style.use("ggplot")
np.random.seed(123)

# 目標分布
a, b = 1.5, 2.0
x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 100)  # ベータ分布x=0.001-0.999まで100個準備
plt.plot(x, beta.pdf(x, a, b))

# 上記ベータ分布の最大値のxを求める
f = beta(a=a, b=b).pdf
res = opt.fmin(lambda x: -f(x), 0.3)  # 最大値求めるのを最小値求めるのに変えるために-f(x)にしている
y_max = f(res)

y_max

NMCS = 5000
x_mcs = uniform.rvs(size=NMCS)  # uniform.rvs:一様分布に従うサンプリング
r = uniform.rvs(size=NMCS) * y_max
accept = x_mcs[r <= f(x_mcs)]
plt.hist(accept, bins=30, rwidth=0.8, label="rejection sampling")
x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 100)
plt.plot(x, beta.pdf(x, a, b), label="Target dis")
plt.legend()


