# ポケモンを題材に因果推論を実践してみる
# https://tepppei.hatenablog.com/

## モンスターボールとスーパーボールの捕まえやすさに有意差あるか検定 ##
# csvは https://github.com/Teppei-Kanayama/pokemon_get_simulator/blob/master/resources/data.csv
df <- read.table("pokemon_ball_data.csv",
                header=T, 
                sep=",",
                fileEncoding="UTF-8")

# サンプリングされたモンスターボールとスーパーボール投げた数
m_throwns <- df[df$ball_type=='monster', ]$thrown_balls
s_throwns <- df[df$ball_type=='super', ]$thrown_balls
print(paste("length(m_throwns):", length(m_throwns)))
print(paste("length(s_throwns):", length(s_throwns)))

# 平均値,不変分散確認
print(paste("m_throwns mean var:", mean(m_throwns), var(m_throwns)))
print(paste("s_throwns mean var:", mean(s_throwns), var(s_throwns)))

# 2群が等分散かF検定 帰無仮説（2群の分散に差がない）
var.test(x=m_throwns, y=s_throwns, conf.level=0.95)  
# p値<0.05なので、帰無仮説棄却=2群の分散に差がある

# 対応のない、ウェルチのt検定（不等分散のt検定） 帰無仮説（2群の平均に差がない）
t.test(x=m_throwns, y=s_throwns, conf.level=0.95, var.equal=F, paired=F)
# p値>0.05なので、帰無仮説棄却できない=平均の差がない
# スーパーボールの方が性能いいとはならなかった
# 直観と違うのでセレクションバイアスの可能性ある


## 交絡因子の存在 ##
# サンプルを分析した結果、
# 「捕まえやすそうなポケモンにはモンスターボールを使い、捕まえにくそうなポケモンにはスーパーボールを使っている」
# つまり、ランダムサンプリングではないことが判定した。
# 先ほどのt検定はセレクションバイアスによって歪んでいる可能性が出てきた。

# サンプルを分析した結果から、
# 「どのボールを使うか」（目的変数）は「捕まえるのに使ったボールの個数」（説明変数）にのみ依存する前提だったが、
# 「ポケモンのつかまえやすさ」の変数が上記2変数の両方に対して相関を持つ可能性が高いことがわかる。
# このような 目的変数と説明変数どちらにも相関がある変数を交絡因子と呼び、
# 交絡因子を無視して効果検証を行うと、セレクションバイアスにより実際の効果と異なる結果が出てしまう。

# 「ポケモンのつかまえやすさ」の影響を取り除くために線形重回帰を用いたモデル化を行いたうえで、
# 改めて「スーパーボールは本当にモンスターボールより捕まえやすいのか？」という仮説を検証する
# ここからはpythonで回帰モデルつくる 




