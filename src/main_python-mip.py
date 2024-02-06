from mip import Model, maximize, minimize, xsum

m = Model()  # 数理モデル
# 変数
x = m.add_var("x")
y = m.add_var("y")
# 目的関数
m.objective = maximize(100 * x + 100 * y)
# 制約条件
m += x + 2 * y <= 16  # 材料Aの上限
m += 3 * x + y <= 18  # 材料Bの上限
m.optimize()  # ソルバーの実行
print(x.x, y.x)