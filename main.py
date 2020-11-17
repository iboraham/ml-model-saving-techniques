from test import run_test

res = run_test()
res['dif'] = res['joblib'] - res['pickle']

# Test multiple times
for i in range(100):
    res.append(run_test())
res['dif'] = res['joblib'] - res['pickle']

res.groupby(res.index).agg({'mean','std'}).to_csv('results.csv')
