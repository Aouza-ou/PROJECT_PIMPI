import pyarrow.parquet as pq

Path_data = r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet"
# Read only the metadata (takes less than a second)
metadata = pq.read_metadata(Path_data)

# Print the exact number of rows and columns
print(f"Total Rows: {metadata.num_rows}")
print(f"Total Columns: {metadata.num_columns}")
