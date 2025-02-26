import pandas as pd
import re
import seaborn as sns

data = sns.load_dataset('titanic')
col = input("Enter column (name, age, sex, survived, pclass): ")
pat = input(f"Enter pattern for '{col}': ")

if col not in data.columns:
    print("Invalid column name.")
else:
    matches = [v for v in data[col].dropna().astype(str) if re.match(pat, v)]
    print(f"Matches ({len(matches)}):", matches if matches else "No matches found.")