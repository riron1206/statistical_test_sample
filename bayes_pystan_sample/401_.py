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

import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
# %matplotlib inline

plt.style.use("ggplot")

p_a = 3.0 / 10.0
p_b = 5.0 / 9.0
p_prior = 0.5
#0:blue, 1:red
data = [0,1,0,0,1,1,1]

N_data = 7
likehood_a = bernoulli.pmf(data[:N_data], p_a)
likehood_b = bernoulli.pmf(data[:N_data], p_b)

likehood_a

pa_posterior = p_prior  # 事前分布
pb_posterior = p_prior
pa_posterior *= np.prod(likehood_a)  # 積計算
pb_posterior *= np.prod(likehood_b)
norm = pa_posterior + pb_posterior  # エビデンス（規格化）
df = pd.DataFrame([pa_posterior/norm, pb_posterior/norm], columns=["post"])  # 事後分布の確率分布
x = np.arange(df.shape[0])
plt.bar(x,df["post"])
plt.xticks(x,["a","b"])



df




