#hello_world.py
## Initial file for DS/ML projects

import pandas as pd

df = pd.read_csv("data/iris.csv")

print(f"Shape: {df.shape}")
print(df)
