import pandas as pd
import numpy as np


x_vals = np.linspace(0, 2 * np.pi, num=10)
y_vals = np.linspace(0, 2 * np.pi, num=20)

data = np.random.rand(x_vals.size, y_vals.size)

df = pd.DataFrame(data, index=x_vals, columns=y_vals)
print(df)
df.to_csv("test.csv")
df.to_csv("test.csv")

test_df = pd.read_csv("test.csv", index_col=0)
print(test_df)