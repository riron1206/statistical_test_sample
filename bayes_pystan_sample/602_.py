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

# # 不動産価格の単回帰

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystan
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/real_estate1.xlsx")

# +
# id, space, room, value(予測する物件価格)

df.head()

# +
# 物件価格は部屋の大きさに正の相関あり=単回帰でモデリングできそう
# また物件価格のヒストグラムは正規分布と仮定

plt.scatter(df["space"],df["value"])

# +
# PyStanは書き方が特殊
# コメントで以下のようなブロックを書く必要がある
# ①dataブロック（観測したデータの入れ物）→
# ②パラメータのブロック→
#  平均値が単回帰になる正規分布と仮定してるからパラメータは、傾き:a, 切片:b, 標準偏差:σ
# ③統計モデルの（尤度関数）ブロック→
# ④事前分布
#  省略している。事前分布していなければ無条件兼事前分布が事前分布になる

stan_model = """
data {
  int N;
  real X[N];
  real Y[N];
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

"""

# +
# PyStanはコンパイルが必要（C++でコンパイルされる）
# 結構時間かかる

sm = pystan.StanModel(model_code = stan_model)

# +
# dataブロックに入れるデータを辞書型で渡す

stan_data = {"N":df.shape[0], "X":df["space"],"Y":df["value"]}

# +
# MCMCでサンプリング

fit = sm.sampling(data = stan_data, iter=2000, warmup=500, chains = 3, seed=123)

# +
# 結果の抽出
# 事後分布の平均値や誤差が表示される
# Rhat<=1ならうまく収束している
# a=77.99なので、1平米上がると価格これぐらい上がる

fit

# +
# 事後分布plot
# トレースプロットは横軸step縦軸サンプリングの値
# トレースプロットがまんべんなくplot（サンプリング）されているなら収束してる

fig = fit.plot()

# +
# 推定したパラメータで単回帰の式構築
# 正規分布を仮定しているから観測テータの平均の点を通るような式になってることがわかる

a = 77.99
b = -692.9
x = np.arange(40,90,1)
y = a * x + b
plt.plot(x,y)
plt.scatter(df["space"],df["value"])
# -


