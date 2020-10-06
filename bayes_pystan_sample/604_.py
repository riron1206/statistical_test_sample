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

# # 薬の致死量をジスティクス回帰で調べる

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystan
from scipy.stats import mstats
# %matplotlib inline

plt.style.use("ggplot")

df = pd.read_excel("./data/dose_response.xlsx")

# +
# log10 C: 薬品の投与量の対数
# death: １なら死んでる0なら生存

df.head()

# +
# 薬品の量がある閾値を超えると死亡確率急激に増加する

plt.scatter(df["log10 C"], df["death"])

# +
# PyStanは書き方が特殊
# コメントで以下のようなブロックを書く必要がある
# ①dataブロック（観測したデータの入れ物）→
# ②パラメータのブロック→
#  ベルヌーイロジットののパラメータはa,bだけ
# ③統計モデルの（尤度関数）ブロック→
# ④事前分布
#  省略している。事前分布していなければ無条件兼事前分布が事前分布になる

stan_model = """
data {
  int N;
  real X[N];
  int<lower=0, upper=1> Y[N];
}

parameters {
  real a;
  real b;
}

model {
  for (n in 1:N){
    Y[n] ~ bernoulli_logit(a * X[n] + b);
  }
}

"""

# +
# PyStanはコンパイルが必要（C++でコンパイルされる）
# 結構時間かかる

sm = pystan.StanModel(model_code= stan_model)

# +
# dataブロックに入れるデータを辞書型で渡す

stan_data = {"N":df.shape[0], "X":df["log10 C"], "Y":df["death"]}

# +
# MCMCでサンプリング

fit = sm.sampling(data = stan_data, iter = 2000, warmup=500, chains=3, seed=123)

# +
# 結果の抽出
# 事後分布の平均値や誤差が表示される
# Rhat<=1ならうまく収束している

fit

# +
# 推定されたパラメータ

a, b = 13.57, -20.27

# +
# 事後分布plot
# トレースプロットは横軸step縦軸サンプリングの値
# トレースプロットがまんべんなくplot（サンプリング）されているなら収束してる

fig = fit.plot()

# +
# ベイズ信頼区間を出すためにパラメータの値抽出

ms_a = fit.extract("a")["a"]
ms_b = fit.extract("b")["b"]

# +
# 抽出したパラメータでロジスティックス関数何本も書く

x = np.arange(1.0, 2.0, 0.01)
f = lambda x : 1.0 / (1.0 + np.exp(-x))  # ロジスティックス関数
df_b = pd.DataFrame([])
for i in range(x.shape[0]):
    df_b[i] = f(ms_a * x[i] + ms_b)
# -

df_b.head()

# +
# ベイズ信頼区間の上限下限

low_y50, high_y50 = mstats.mquantiles(df_b, [0.25, 0.75], axis=0)
low_y95, high_y95 = mstats.mquantiles(df_b, [0.025, 0.975], axis=0)

# +
# ベイズ信頼区間含めて観測データplot

plt.scatter(df["log10 C"], df["death"])
plt.fill_between(x, low_y95, high_y95, alpha = 0.3, color = "gray")
plt.fill_between(x, low_y50, high_y50, alpha = 0.6, color = "darkgray")
plt.plot(x, f(a*x+b))
# -


