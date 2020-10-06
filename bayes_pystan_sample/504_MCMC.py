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
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

plt.style.use("ggplot")
np.random.seed(123)

p_trans = np.zeros([3,3])

# 遷移核：どの状態にあるかの確率
# 0:office, 1:kyukei, 2:jikken
p_trans[0, 0] = 0.1
p_trans[0, 1] = 0.2
p_trans[0, 2] = 0.7
p_trans[1, 0] = 0.1
p_trans[1, 1] = 0.4
p_trans[1, 2] = 0.5
p_trans[2, 0] = 0.3
p_trans[2, 1] = 0.3
p_trans[2, 2] = 0.4

p_trans

NMCS = 400  # モンテカルロステップ数（試行回数）
c_state = 0  # 初期状態
c_arr = [c_state]
for i in range(NMCS):
    # np.random.choice(3, 1 なら0,1,2のどれかを選択、pで確率指定
    current = np.random.choice(3, 1, p=p_trans[c_state, :]) 
    c_state = current[0]
    c_arr.append(c_state)
df = pd.DataFrame(c_arr)  # モンテカルロステップごとの状態をデータフレームに

c_state = 0
np.random.choice(3, 1, p=p_trans[c_state, :])

df.head()

plt.plot(df[0])
plt.xlabel("MCS")
plt.ylabel("place")

# ステップ数変えても分布変わらない。定常状態
plt.hist([df[0][:10], df[0][:50], df[0][:200], df[0][:400]], label=["10","50","200","400"])
plt.xlabel("place")
plt.ylabel("frequency")
plt.legend()


