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
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/MAP_sample.xlsx", index_col="id")

df.head()

plt.hist(df["value"])


def likelihood(mu, *args):
    """尤度関数（本来は尤度関数の確率の積だが、計算簡単にするために-log10取って和をとってる）"""
    # stats.norm.pdfは正規分布確の率密度関数
    # norm.pdf(x=1.0, loc=0, scale=1) なら、期待値loc，標準偏差scaleの正規分布の確率密度関数のx=1.0での値を取得
    # 推定する事後分布のパラメータ=平均値:μ, argsは事後分布の期待値（実際のデータのこと）として尤度関数（確率値）計算
    li = -np.log10(stats.norm.pdf(mu, loc=args))
    return(np.sum(li))


# 尤度関数を最小化する事後分布のパラメータ=平均値:μ。初期値=1として実行
optimize.minimize(likelihood, 1, args=df["value"])

# x: array([4.97609903])が事後分布のパラメータ=平均値:μ  
# ヒストグラムでも平均値は5付近なので合ってそう
