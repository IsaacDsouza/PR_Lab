import math
import pandas as pd
from collections import Counter

def shannon_entropy(data):
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in Counter(data).values())

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
col = input(f"Select column {df.columns.tolist()}: ")
print(f"Shannon Entropy of '{col}': {shannon_entropy(df[col])}" if col in df else "Invalid column.")
