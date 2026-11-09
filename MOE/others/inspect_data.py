import pandas as pd
import numpy as np
import pyarrow.parquet as pq  #  Changed import to access ParquetFile directly

file_path = r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet"

print(" Safely loading first block of data (Zero RAM crash risk)...")

#  SAFE METHOD: Read only the first "Row Group" (block) instead of the whole file
parquet_file = pq.ParquetFile(file_path)
df_sample = parquet_file.read_row_group(0).to_pandas().head(1)

print("\n=== COLUMN NAMES ===")
print(df_sample.columns.tolist())

print("\n=== DATA TYPES ===")
print(df_sample.dtypes)

print("\n=== FIRST ROW PREVIEW ===")
first_row = df_sample.iloc[0]

for col in df_sample.columns:
    val = first_row[col]
    print(f"\nColumn: {col}")
    print(f"Type: {type(val)}")
    
    # If it's a binary string/bytes
    if isinstance(val, bytes):
        print(f"Length of bytes: {len(val)}")
        print(f"Sample bytes (hex): {val[:20].hex()}...")
        
    # If it's an array/list
    elif isinstance(val, (list, np.ndarray)):
        print(f"Length/Shape: {len(val)}")
        print(f"Sample data: {val[:3]}")
        
    # If it's a Pandas Series (sometimes happens with nested parquet data)
    elif isinstance(val, pd.Series):
        print(f"Length: {len(val)}")
        print(f"Sample data: {val.head(3).tolist()}")
        
    else:
        print(f"Value: {val}")