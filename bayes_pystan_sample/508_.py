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
from scipy.stats import norm, gamma  # 正規分布とガンマ分布
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

plt.style.use("ggplot")
np.random.seed(123)

k = 11  # ガンマ分布のパラメータの初期値
lam = 1  # ガンマ分布のパラメータの初期値
x = np.linspace(gamma.ppf(0.001,k), gamma.ppf(0.999,k),100)
plt.plot(x, gamma.pdf(x, k, scale=1/lam))

f = lambda x, lam, k : -lam + (k-1) / x


def leap_flog(q,p,eps,lam,k,f):
    """リープフロッグ法"""
    p_new = p + 0.5 * eps * f(q, lam, k)
    q_new = q + eps * p_new
    p_new = p_new + 0.5 * eps * f(q_new,lam, k)
    q, p = q_new, p_new
    return q, p


# +
# ハミルトニアンモンテカルロ法

eps = 1e-2
q, p = 4.0, 0.0
L = 100
NMCS = 10000
warmup = 5000  # ウォームアップ
lf_arr = np.zeros([NMCS,2])
n_accept = 0
for mcs in range(NMCS):
    hamiltonian_c = 0.5 * p ** 2 + lam * q - (k-1) * np.log(q)
    q_c, p_c = q, p 
    for i in range(L):
        q_c, p_c = leap_flog(q_c, p_c, eps, lam, k, f)
    hamiltonian_new = 0.5 * p_c ** 2 + lam * q_c - (k-1) * np.log(q_c)  # 現在のハミルトニアン
    if np.random.rand() < np.exp(hamiltonian_c - hamiltonian_new):
        q, p = q_c, p_c  # 更新するq, p
        hamiltonian_c = hamiltonian_new
        n_accept += 1
    lf_arr[mcs,:] = q,p
    p = norm.rvs()  # 別の乱数発生
df = pd.DataFrame(lf_arr[warmup:], columns=["q","p"])  # ウォームアップ分は捨てる
plt.figure(figsize=(9,6))
x, y = np.linspace(0, 20, 100), np.linspace(-3,3, 100)
X, Y = np.meshgrid(x,y)
CS = plt.contour(X, Y, 0.5 * Y ** 2 - (k-1)*np.log(X) + lam * X, levels = [-12, -10, -5, 0])
plt.clabel(CS, inline=1, fontsize=10)
plt.scatter(df["q"][0],df["p"][0])
plt.scatter(df["q"],df["p"], marker=".", c = df.index, cmap="Blues")
plt.colorbar()
print("accept ratio: ", n_accept / NMCS)
# -

df["q"].hist(bins=40)
x = np.linspace(gamma.ppf(0.001,k), gamma.ppf(0.999,k),100)
plt.plot(x, gamma.pdf(x, k, scale=1/lam))


