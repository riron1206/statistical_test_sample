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
from scipy.special import gamma  # ガンマ関数
# %matplotlib inline

plt.style.use("ggplot")
np.random.seed(123)

# 次元の呪いにより、高次元の場合、モンテカルロの解と解析解が一致しなくなる
# 次元上がるとランダムに点を打つだけでは成立しないことわかる
accept_dict = {}
NMCS = 40000  # サンプル数
Ndim = 15  # 次元数
for ii in range(Ndim):
    print(f"\n---- dim={ii}----")
    accept = 0  # 初期化
    for i in range(NMCS):
        x = 2 * np.random.rand(ii) - 1.0  # 座標
        r = (np.sum(x**2)) ** 0.5  # 距離
        if r <= 1:
            accept += 1  # 棄却サンプリング
    accept_ratio = accept / NMCS  # 受容確率
    analytical = np.pi **(ii/2.0) / (2 ** ii * gamma(ii / 2 + 1))  # N次元の級の体積=解析解
    print("accept ratio: ", accept_ratio)  # モンテカルロで出した結果
    print("analytical solution: ", analytical)  # 解析解
    accept_dict.update({ii:accept_ratio / analytical})

df = pd.DataFrame.from_dict(accept_dict, orient="index")
plt.scatter(df.index,df[0])
plt.xlabel("dimension")
plt.ylabel("ratio[-]")


