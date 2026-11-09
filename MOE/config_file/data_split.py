import pandas as pd 

df = pd.read_parquet(r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet")
n_val = int(0.05 * len(df))

df.iloc[:-n_val].to_parquet(r"D:\PROJECT_PIMPI\MOE\DATA\train_core.parquet")
df.iloc[:-n_val].to_parquet(r"D:\PROJECT_PIMPI\MOE\DATA\val_fixed.parquet")

print(f"train_core: {len(df)-n_val} row | val_fixed: {n_val} rows")