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

# # 不動産の価格を複数の説明変数から予測する

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystan
import seaborn as sns
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/real_estate2.xlsx")

df.head()

df["elapsed"] = 2018 - df["year"]

df["distance"].unique()

dis_arr = df["distance"].unique()

dis_arr

dis_dict = {dis_arr[0]:10, dis_arr[1]:15, dis_arr[2]:5, dis_arr[3]:20, dis_arr[4]:30, dis_arr[5]:np.nan}

dis_dict

df["distance2"] = df["distance"].apply(lambda x:dis_dict[x])

df.head()

# +
# 欠損行削除

df = df.dropna()

# +
# 数値列のみ使う

df2 = df[["space","elapsed","distance2","value"]]
# -

df2.head()

# +
# 築年数が価格に強く影響する

g = sns.PairGrid(df2)
g = g.map_lower(sns.kdeplot)  # 左下の要素は等高線
g = g.map_diag(sns.distplot, kde=False)  # 対角成分はカーネル密度推定なしのヒストグラム
g = g.map_upper(plt.scatter)   # 右上の要素は散布図

# +
# 物件価格は
# 駅からの距離
# 部屋の大きさ
# 築年数 の線形結合で表せるとする
# 予測値のヒストグラムは正規分布すると仮定する

stan_model = """
data {
  int N;
  real elapsed[N];
  real dis[N];
  real space[N];
  real Y[N];
}

parameters {
  real d;
  real s;
  real e;
  real b;
  real<lower=0> sigma;
}

model {
  real mu;
  for (n in 1:N){
    mu = e * elapsed[n] + d * dis[n] + s * space[n] + b;
    Y[n] ~ normal(mu, sigma);
  }
}

"""
# -

sm = pystan.StanModel(model_code=stan_model)

stan_data = {"N":df.shape[0],"elapsed":df["elapsed"],"dis":df["distance2"], "space":df["space"], "Y":df["value"]}

fit = sm.sampling(data = stan_data, iter=2000, warmup=500, chains=4, seed=123)

fit

fig = fit.plot()


