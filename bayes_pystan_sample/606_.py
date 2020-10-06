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

# # 身長の伸び具合を個人差を含めて予測する
# - 単純な単回帰で扱わないのは、身長の伸び具合は個人差があるはずなので
# - y[i]（個人ごとの身長） = a[i]（個人ごとの傾き） * x(年齢) + b[i]
# - a[i]（個人ごと傾き） = a0（共通部分）+ a_id(個人ごとのばらつき)
# - b[i]（個人ごとの切片） = b0（共通部分） + b_id（個人ごとのばらつき）
# - yとbは正規分布していると仮定する
# - 階層とついてるのは、個人ごとのばらつきの値a_id, b_idを推定してから、その値を含めて身長yの値を推定しているため

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats
import pystan
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/multilevel_modeling.xlsx")

# +
# age: 年齢
# height: 身長

df.head()
# -

df["id"].unique()

groups = df.groupby(df["id"])

# +
# idごとの年齢と身長の関係
# 人によって身長の伸び具合が違うのがわかる

plt.figure(figsize=(9,9))
for name, group in groups:
    plt.plot(group["age"],group["height"],label=name)
plt.legend()

# +
# PyStanは書き方が特殊
# コメントで以下のようなブロックを書く必要がある
# ①dataブロック（観測したデータの入れ物）→
# ②パラメータのブロック→
#  ベルヌーイロジットののパラメータはa,bだけ
# ③統計モデルの（尤度関数）ブロック→
# ④事前分布
#  省略している。事前分布していなければ無条件兼事前分布が事前分布になる

# パラメータのs_a, s_b, s_Yは正規分布の標準偏差

# 階層ベイズモデルはtransformed parameters { でパラメータを組み合わせる

stan_model = """
data {
  int N;
  int N_id;
  real X[N];
  real Y[N];
  int<lower=1, upper=N_id> s_id[N];
}

parameters {
  real a0;
  real b0;
  real a_id[N_id];
  real b_id[N_id];
  real<lower=0> s_a;
  real<lower=0> s_b;
  real<lower=0> s_Y;
}

transformed parameters {
  real a[N_id];
  real b[N_id];
  for (n in 1:N_id){
    a[n] = a0 + a_id[n];
    b[n] = b0 + b_id[n];
  }
}

model {
  for (id in 1:N_id){
    a_id[id] ~ normal(0, s_a);
    b_id[id] ~ normal(0, s_b);
  }
  
  for (n in 1:N){
    Y[n] ~ normal(a[s_id[n]] * X[n] + b[s_id[n]], s_Y);
  }
}

"""
# -

sm = pystan.StanModel(model_code=stan_model)

stan_data = {"N":df.shape[0], "N_id":15, "X":df["age"], "Y":df["height"], "s_id":df["id"]}

fit = sm.sampling(data = stan_data, iter=3000, warmup=1000, chains=3, seed=123)

fit

fig = fit.plot()

# +
# ベイズ信頼区間を出すためにパラメータ取り出す

ms_a = fit.extract("a")["a"]
ms_b = fit.extract("b")["b"]

# +
# ms_a[:,0]で出席番号1番の人だけの結果を出す

x = np.arange(18)
df_b = pd.DataFrame([])
for i in range(18):
    df_b[i] = ms_a[:,0] * x[i] + ms_b[:,0]

# +
# 信頼区間計算

low_y50, high_y50 = mstats.mquantiles(df_b, [0.25, 0.75], axis=0)
low_y95, high_y95 = mstats.mquantiles(df_b, [0.025, 0.975], axis=0)

# +
# 出席番号1番の人だけの結果を出す

df_0 = groups.get_group(1)
# -

df_0.head()

# +
# ベイズ信頼区間を塗りつぶす
# 出席番号1の人の身長の推定はすごい綺麗に乗ってる
# これは共通部分の項があるため。共通部分は他のidのデータも使ってモデル作った。データの数が増えたので精度上がった
# 階層ベイズモデルはデータが少ない要素も共通部分によりうまく推定できるところがメリット

plt.plot(df_0["age"],df_0["height"])
plt.fill_between(x, low_y50, high_y50, alpha=0.6, color="darkgray")
plt.fill_between(x, low_y95, high_y95, alpha=0.3, color="gray")
# -


