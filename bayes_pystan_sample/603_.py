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

# # 単回帰でベイズ信頼区間・予測区間確認
# - ベイズ推定は有限のデータからパラメータ推定しているので誤差が伴う
# - 前の単回帰で出した直線なら、傾きや切片が少し違う式があり得る。予測直線にばらつきがあるはず
# - このばらつきがどれぐらいの範囲に収まっているかがベイズ信頼区間
# - ベイズ信頼区間はモデルの精度と同義
# - 予測区間は予測したパラメータを使ったモデルでMCMCサンプリングしたときの値。要は実際の予測値

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystan
from scipy.stats import mstats
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/real_estate1.xlsx")

# +
# generated quantitiesブロックは予測区間出すために必要なブロック
#  = normal_rng( 使うのがgenerated quantities 特有のルール
# generated quantities のデータ入れ物はdataブロックのN_s, X_s

stan_model = """
data {
  int N;
  real X[N];
  real Y[N];
  int N_s;
  real X_s[N_s];
}

parameters {
  real a;
  real b;
  real<lower=0> sigma;
}

model {
  for (n in 1:N){
    Y[n] ~ normal(a * X[n] + b, sigma);
  }
}

generated quantities {
  real Y_s[N_s];
  for (n in 1:N_s){
    Y_s[n] = normal_rng(a * X_s[n] + b, sigma);
  }
}

"""

# +
# コンパイル

sm = pystan.StanModel(model_code=stan_model)

# +
# 予測区間出すためのデータも用意

X_s = np.arange(40,90,1)
N_s = X_s.shape[0]
stan_data = {"N":df.shape[0],"X":df["space"],"Y":df["value"],"N_s":N_s,"X_s":X_s}

# +
# MCMCサンプリング実行

fit = sm.sampling(data = stan_data, iter = 2000, warmup= 500, chains= 3, seed=123)

# +
# extractで推定したパラメータaのデータの抽出を行う

fit.extract("a")

# +
# 辞書型だから値だけとって

ms_a = fit.extract("a")["a"]
# -

ms_a

# +
# plotするとパラメータaのヒストグラムは正規分布であることわかる

plt.hist(ms_a)
# -

ms_b = fit.extract("b")["b"]

# +
# 抽出したパラメーから値作る

df_b = pd.DataFrame([])
for i in range(40, 90, 1):
    df_b[i] = ms_a * i + ms_b
# -

df_b

# +
# ベイズ信頼区間の上限値下限値を出す

low_y50, high_y50 = mstats.mquantiles(df_b, [0.25,0.75], axis=0)
low_y95, high_y95 = mstats.mquantiles(df_b, [0.025,0.975], axis=0)

# +
# ベイズ信頼区間を塗りつぶす
# データが多いところは信頼区間が狭い=予測精度が高い
# データが少ないと信頼区間は広くなる

plt.scatter(df["space"],df["value"])
plt.fill_between(X_s, low_y50, high_y50, alpha=0.6, color="darkgray")
plt.fill_between(X_s, low_y95, high_y95, alpha=0.3, color="gray")
a = 78.3
b = -713.7
y = a * X_s + b
plt.plot(X_s, y, color ="black")

# +
# 予測区間出すために作成した統計モデルの予測値をサンプリング

Y_p = fit.extract("Y_s")["Y_s"]

# +
# 95%信頼区間準備

low_y, high_y = mstats.mquantiles(Y_p, [0.025,0.975], axis=0)

# +
# 予測信頼区間を塗りつぶす
# 95%信頼区間なのでplot点をすっぽり覆うようになってる

plt.scatter(df["space"],df["value"])
plt.fill_between(X_s, low_y, high_y, alpha=0.3, color="gray")
a = 78.3
b = -713.7
y = a * X_s + b
plt.plot(X_s, y, color ="black")
# -


