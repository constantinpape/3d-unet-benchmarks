import pandas as pd
df = pd.read_csv('./results-embl.csv')
md = df.to_markdown(index=False)
print(md)
