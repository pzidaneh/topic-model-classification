import pandas as pd

x = pd.DataFrame([2, 3, 4], columns=['a'])
x['b'] = pd.Series([2, 4])
print(x)
