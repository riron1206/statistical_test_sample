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

# # 気温の時系列データから地球温暖化を予測する
# - x:時間, y:気温の相対値
# - 気温は圧変化幅で推移している
# - 時系列データには状態空間モデルが有効
# - 時系列データは何かしらの法則に従って変化している。完全なランダムは予測できないので注意
# - 法則は前の状態から徐々に変化しているとか（自己回帰）
#
# <br>
#
# ## 状態空間モデル
# ### 次の状態 = 今の状態 + 法則性の変化
# ### u[t+1] = u[t] + e(t)
#
# <br>
#
# ### 今回の仮説
# - 測定気温は正規分布してる。測定値は誤差があるはずなのでその分を正規分布で仮定
# - u[n] = u[n-1] + e[n]
# - e[n] ~ Normal(0, σ_u)
# - y[n] ~ Normal(u[n], σ)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats
import pystan
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/temperature_series.xlsx")

df.head()

# +
# 時間たつと相対値の気温上がってそう

plt.plot(df["x"],df["y"])

# +
# PyStanは書き方が特殊
# コメントで以下のようなブロックを書く必要がある
# ①dataブロック（観測したデータの入れ物）→
#  T_newは一期先の気温
# ②パラメータのブロック→
# ③統計モデルの（尤度関数）ブロック→
#  前の状態から次の状態が決まるのでmu[t] ~ normal(mu[t-1], s_mu);
#  想定誤差入れtるからY[t] ~ normal(mu[t], s_Y);
# ④事前分布
#  省略している。事前分布していなければ無条件兼事前分布が事前分布になる

# 予測するためにgenerated quantities {つける

stan_model = """
data {
  int T;
  int T_new;
  real Y[T];
}

parameters {
  real mu[T];
  real<lower=0> s_mu;
  real<lower=0> s_Y;
}

model {
  for (t in 2:T){
    mu[t] ~ normal(mu[t-1], s_mu);
  }
  
  for (t in 1:T){
    Y[t] ~ normal(mu[t], s_Y);
  }
}

generated quantities {
  real mu_new;
  real Y_new[T+T_new];
  for (t in 1:T){
    Y_new[t] = normal_rng(mu[t], s_Y);
  }
  mu_new = normal_rng(mu[T], s_mu);
  Y_new[T+T_new] = normal_rng(mu_new, s_Y);
}

"""
# -

sm = pystan.StanModel(model_code=stan_model)

stan_data = {"T":df.shape[0],"T_new":1, "Y":df["y"]}

fit = sm.sampling(data = stan_data, iter=3000, warmup=1500, seed=123, chains=3)

fit

fig = fit.plot()

# +
# 予測値取り出す

Y_new_arr = fit.extract("Y_new")["Y_new"]

# +
# ベイズ信頼区間

low_y50, high_y50 = mstats.mquantiles(Y_new_arr, [0.25, 0.75], axis=0)
low_y95, high_y95 = mstats.mquantiles(Y_new_arr, [0.025, 0.975], axis=0)
# -

plt.plot(df["x"],df["y"])
x = df["x"].values
x = np.append(x, 2017)
plt.fill_between(x, low_y50, high_y50, alpha=0.6, color="darkgray")
plt.fill_between(x, low_y95, high_y95, alpha=0.3, color="gray")


