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

# # ギブスサンプラー
# - MCMCのアルゴリズムの一つ
# - 要は、仮定した確率分布から、詳細つり合いの条件を満たす分布の値をサンプリングするための1手法
#     - 他に、ギブスサンプラー や ハミルトニアンんモンテカルロ などがある
#     
# <br>
#
# - 多変量正規分布で考えている
# - ギブスサンプリングは確率変数を交互に決めてやる

# +

import numpy as np
from scipy.stats import norm, multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from IPython.html.widgets import interact
# %matplotlib inline
# -

plt.style.use("ggplot")
np.random.seed(123)

mu = np.array([1, 2])
cov = np.array([[1.0, 0.7],[0.7, 1.0]])

x = np.arange(-2, 4, 0.01)
y = np.arange(-1, 5, 0.01)
X, Y = np.meshgrid(x,y)
pos = np.zeros([X.shape[0],X.shape[1],2])
pos[:,:,0] = X
pos[:,:,1] = Y

rv = multivariate_normal(mu, cov)
plt.figure(figsize=(4,4))
plt.contour(X, Y, rv.pdf(pos), cmap="plasma")
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")

NMCS = 500
sigma12 = cov[1,0]
sigma1 = cov[0,0] ** 0.5
sigma2 = cov[1,1] ** 0.5
rho = sigma12 / (sigma1*sigma2)
z1 = 0.0
z2 = 0.0
z1_mcs = [z1]
z2_mcs = [z2]
for i in range(NMCS):
    #f(z1|z2)
    z1 = norm.rvs(loc=rho*z2, scale=(1-rho**2)**0.5)  # z2固定してz1を正規乱数で決める
    z1_mcs.append(z1)
    z2_mcs.append(z2)
    #f(z2|z1)
    z2 = norm.rvs(loc=rho*z1, scale=(1-rho**2)**0.5)  # z1固定してz2を正規乱数で決める
    z1_mcs.append(z1)
    z2_mcs.append(z2)
df1 = pd.DataFrame(mu[0] + np.array(z1_mcs) * sigma1)    
df2 = pd.DataFrame(mu[1] + np.array(z2_mcs) * sigma2)    

df1.head()

plt.plot(df1[0],df2[0])


@interact(mcs=(0,100,1))
def animation(mcs=0):
    plt.contour(X, Y, rv.pdf(pos), cmap="plasma")
    plt.xlim([-3.0, 4.5])
    plt.ylim([-1.0, 5.0])
    plt.plot(df1[0][:mcs], df2[0][:mcs])
    plt.show()


