import os 
import pyarrow.parquet as pq
import pandas as pd 
from pathlib import Path

#from .DATA import DATA

def split_data_rows(input_file: str, output_dir: str = "."):
	print(f"Original file: {input_file} and inspecting metadata...")

	# 1. Read metadata instantly without Loading rows into RAM
	parquet_file = pq.ParquetFile(input_file)
	total_rows = parquet_file.metadata.num_rows
	midpoint = total_rows // 2

	print(f"Total Rows Found: {total_rows:,}")
	print(f"Midpoint calculated at row: {midpoint:,}")

	# 2. Load the complete data table efficiently
	print("Loading full table into memory for row slicing...")
	table = parquet_file.read()

	# 3. Slice row-wise (part 1: top half, part 2: bottom half)
	print("Slicing dataset row-wise (50/50)...")
	data_table1 = table.slice(offset=0, length=midpoint)
	data_table2 = table.slice(offset=midpoint, length=total_rows - midpoint)

	# Define the paths for the output files
	#path_Dict = r"D:\PROJECT_PIMPI\MOE\DATA"
	out_path = Path(input_file).parent # Use the same directory as the input file
	data1 = out_path / "train_part_1.parquet"
	data2 = out_path / "train_part_2.parquet"

	# 4. save the two parts securely back to disk 
	print(f"Saving part 1 ({data_table1.num_rows:,} rows) -> {data1}")
	pq.write_table(data_table1, data1)
	print(f"Saving part 2 ({data_table2.num_rows:,} rows) -> {data2}")
	pq.write_table(data_table2, data2)

	print("\nRow-wise split completed successfully! No SMILES strings were fragmented.")

if __name__ == "__main__":
		#  # Adjust this path if your train.parquet is located inside another folder

		INPUT_PATH =  r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet"

		if os.path.exists(INPUT_PATH):
			split_data_rows(INPUT_PATH)
		else:
			print(f"ERROR: COULD NOT FIND  '{INPUT_PATH}' IN THIS DIRECTORY. CHECK YOUR FILE PATH.")





