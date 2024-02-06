import pulp


# number of products
nP = 4
rP = [i for i in range(nP)]
# number of materials
nM = 3
rM = [i for i in range(nM)]

# material stock
stock = [35, 22, 27]
# material request amount of each product. nM*nP
request_raw = [2, 0, 1, 3, 2, 0, 0, 2, 2, 2, 2, 2]
# multi dimension array of request
request = [[request_raw[m+p*nM] for m in rM] for p in rP]
print('req ', request)
# profit gain of each product
gain = [3, 4, 4, 5]

# 最大化
problem = pulp.LpProblem("sample", pulp.const.LpMaximize)

# variable 決定変数
x = {}
for p in rP:
    # x[p] = pulp.LpVariable(f'production_amount_{p}', cat='Continuous')
    x[p] = pulp.LpVariable(f'production_amount_{p}', cat='Integer')

# constraints 制約条件
for p in rP:
    # all production amount >= 0
    problem += x[p] >= 0
for m in rM:
    # for all material, sum of consumpution <= stock
    consumption = [request[p][m]*x[p] for p in rP]
    problem += pulp.lpSum(consumption) <= stock[m]

# objective 目的関数
problem += pulp.lpSum([gain[p]*x[p] for p in rP])

status = problem.solve()

print('status: ', pulp.LpStatus[status])
for p in rP:
    print(f'p: {p} amount: {x[p].value()}')
print('obj=', problem.objective.value())
