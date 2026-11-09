


import pandas as pd
import numpy as np
import pyarrow.parquet as pq
 

file_path = r"D:\\PROJECT_PIMPI\\MOE\\DATA\\train.parquet"
# 1. Use pyarrow to read the schema instantly (takes 0 seconds)
print("=== COLUMN NAMES & DATA TYPES ===")
parquet_file = pq.ParquetFile(file_path)
print(parquet_file.schema)

# 2. Open the file and read ONLY the columns you actually need
# Replace 'smiles' with whatever your SMILES column is named from the printout above
smiles_column_name = "normalized_smiles" 

print(f"\nAnalyzing length distribution from '{smiles_column_name}' column...")
# We use pyarrow directly to read just one column to keep memory usage low
table = parquet_file.read(columns=[smiles_column_name])
df_lengths = table.to_pandas()

# Calculate length of text strings
df_lengths['char_length'] = df_lengths[smiles_column_name].str.len()

# 3. Calculate key statistics for your model configuration
max_len = df_lengths['char_length'].max()
min_len = df_lengths['char_length'].min()
mean_len = df_lengths['char_length'].mean()
median_len = df_lengths['char_length'].median()

# Find the 99th percentile 
p99_len = int(np.percentile(df_lengths['char_length'], 99))

print("\n=== SMILES LENGTH STATISTICS ===")
print(f"Shortest SMILES: {min_len} characters")
print(f"Average SMILES Length: {mean_len:.1f} characters")
print(f"Median SMILES Length: {median_len} characters")
print(f"99% of molecules are shorter than: {p99_len} characters 🌟 (Use this for padding!)")
print(f"Absolute Longest SMILES: {max_len} characters ⚠️")

outliers = len(df_lengths[df_lengths['char_length'] > p99_len])
print(f"Number of extreme outlier rows above {p99_len} chars: {outliers:,}")

