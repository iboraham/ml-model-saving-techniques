from test import run_test
import pandas as pd

res = run_test()

# Test multiple times
for i in range(100):
    temp = run_test().copy()
    res = pd.concat([res,temp])
res['dif'] = res['joblib'] - res['pickle']
print(res)
res.groupby(res.index).agg({'mean','std'}).to_csv('results.csv')
