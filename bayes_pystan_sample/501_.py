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
import pandas as pd
import matplotlib.pyplot as plt
from IPython.html.widgets import interact
# %matplotlib inline

plt.style.use("ggplot")
np.random.seed(123)

NMC = 100  # 座標の数
xmc = np.random.rand(NMC)  # x軸の座標
ymc = np.random.rand(NMC)  # y軸の座標
r = (xmc ** 2 + ymc ** 2) ** 0.5  # x,y軸上の距離計算
accept = np.where(r<=1, 1, 0)  # 半径1以下に入ったか判定
accept_ratio = np.sum(accept) / NMC  # 何割入ったか
print(accept_ratio)  # 解析解は(pi/4)/1=0.785 なので解析解よりやや多い

NMC = 100
xmc = np.random.rand(NMC)
ymc = np.random.rand(NMC)
@interact(mcs=(0,NMC,1))  # インタラクティブに点の数変更する
def animation(mcs=0):
    """円の中に入る乱数の点をplot"""
    plt.figure(figsize=(6,6))
    plt.xlim([0,1])
    plt.ylim([0,1])
    x = np.arange(0,1,0.001)
    y = (1 - x ** 2) ** 0.5
    y2 = np.ones(x.shape[0])
    plt.plot(x,y)
    plt.fill_between(x, y, alpha=0.3)  # 塗りつぶし。alphaで透過
    plt.fill_between(x, y, y2,alpha=0.3)
    r = (xmc[:mcs] ** 2 + ymc[:mcs] ** 2) ** 0.5
    accept = np.where(r<=1, 1, 0)
    accept_ratio = np.sum(accept) / mcs
    plt.scatter(xmc[:mcs], ymc[:mcs], color="black", marker=".")
    plt.show()
    print("Monte Carlo: ",accept_ratio)
    print("Analytical Solution: ", np.pi / 4.0)


pi_mcs = []
NMC = 2000
xmc = np.random.rand(NMC)
ymc = np.random.rand(NMC)
for mcs in range(1,NMC):
    r = (xmc[:mcs] ** 2 + ymc[:mcs] ** 2) ** 0.5
    accept = np.where(r<=1, 1, 0)
    accept_ratio = np.sum(accept) / mcs
    pi_mcs.append(accept_ratio)

pi_x = np.arange(len(pi_mcs)) + 1

plt.plot(pi_x, pi_mcs)
plt.hlines(0.785, pi_x[0], pi_x[-1], linestyles="dashed")
plt.xlabel("MCS")
plt.ylabel("accept ratio")
# サンプル数多いと解析解に漸近していく


