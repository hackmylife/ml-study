V = {'L1': 0.0, 'L2': 0.0}
new_v = V.copy()

cnt = 0
while True:
    new_v['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    new_v['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    delta = abs(new_v['L1'] - V['L1'])
    delta = max(delta, abs(new_v['L2'] - V['L2']))

    V = new_v.copy()

    cnt += 1
    if delta < 0.0001:
        print(V)
        print(cnt)
        break