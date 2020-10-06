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

# # 正規分布に従うデータでMCMC実行

import pandas as pd
import matplotlib.pyplot as plt
import pystan
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/data.xlsx")

df.head()

# +
# ヒストグラムからデータが正規分布疎っぽいことがわかる
# 尤度を正規分布にしてモデリングしていく

plt.hist(df[0])

# +
# PyStanは書き方が特殊
# コメントで以下のようなブロックを書く必要がある
# ①dataブロック（観測したデータの入れ物）→
# ②パラメータのブロック→
#  real<lower=0> sigma; は標準偏差が正の値に指定
# ③統計モデルの（尤度関数）ブロック→
#  Y[i] ~ normal(mu, sigma); は正規分布
# ④事前分布
#  省略している。事前分布していなければ無条件兼事前分布が事前分布になる

stan_model = """
data {
  int N;
  real Y[N];
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  for (i in 1:N){
    Y[i] ~ normal(mu, sigma);
  }
}
"""

# +
# PyStanはコンパイルが必要（C++でコンパイルされる）
# 結構時間かかる

sm = pystan.StanModel(model_code=stan_model)

# +
# dataブロックに入れるデータを辞書型で渡す

stan_data = {"N":df.shape[0], "Y":df[0]}

# +
# MCMCでサンプリング

fit = sm.sampling(data=stan_data,  # データ渡す
                  iter=2000,  # モンテカルロステップ（サンプリング回数）
                  chains=3,  # 指定したモンテカルロステップ（iter=2000）を何回やるか。3なら2000*3回モンテカルロステップ実行
                  warmup=500,  # ウォームアップ。この分のstepは捨てる
                  seed=123)

# +
# 結果の抽出
# 事後分布の平均値や誤差が表示される
# Rhat<=1ならうまく収束している

fit

# +
# 事後分布plot
# トレースプロットは横軸step縦軸サンプリングの値
# トレースプロットがまんべんなくplot（サンプリング）されているなら収束してる

fig = fit.plot()
# -


