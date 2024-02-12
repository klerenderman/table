import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/home/bruno/Downloads/ccdataset.csv")
df.head()

df.info()
df.dropna()