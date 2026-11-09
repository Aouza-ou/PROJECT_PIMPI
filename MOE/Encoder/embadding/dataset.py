import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from pyarrow import parquet



class SpectralDataset(Dataset):
	def __init__(self, parquet_path: str, is_test: bool = False):
		"""
		Loads the parquet file into memory and serves raw arrays.
		"""

		# Load the parquet file efficiently using pyarrow
		self.df = pd.read_parquet(parquet_path, engine="pyarrow")
		self.is_test  = is_test

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		row = self.df.iloc[idx]

		# 1. Extract raw lists/arrays from the row 
		mzs = np.array(row["ms2_mzs"], dtype=np.float32) 
		intensities = np.array(row["ms2_normalized_intensities"], dtype=np.float32) 

		# ====================================================================
		# LOG-BINNING COMPRESSION (PATH 2)
		# ====================================================================
		NUM_BINS = 2000
		MIN_MZ = 10.0
		MAX_MZ = 3000.0

		# Pre-calculate log boundaries
		log_min = np.log10(MIN_MZ)
		log_max = np.log10(MAX_MZ)

		# Initialize an array of zeros for our fixed bins
		binned_intensities = np.zeros(NUM_BINS, dtype=np.float32)

		if len(mzs) > 0:
			# Keep raw mass values inside our window limits
			clipped_mzs = np.clip(mzs, MIN_MZ, MAX_MZ)

			# Step-by-step log math: Turn masses into bin percentages, then scale to bin IDs
			log_mzs = np.log10(clipped_mzs) 
			bin_indices = ((log_mzs - log_min) / (log_max - log_min) * (NUM_BINS - 1)).astype(np.int32)

			# Safe accumulation: Add intensities together if multiple peaks hit the same bin
			np.add.at(binned_intensities, bin_indices, intensities)

		# Generate the exact centers/edges of our 2,000 bins in log-space for the model to read
		bin_edges = np.logspace(log_min, log_max, NUM_BINS, dtype=np.float32)

		# 2. stack them side-by-side -> Shape: (num_peaks, 2)
		peaks = np.stack([bin_edges, binned_intensities], axis=1) # <-- matrix 
		binned_peaks = peaks
		#peaks_tensor = torch.tensor(peaks, dtype=torch.float32)
		#peaks_tensor = torch.from_numpy(binned_peaks).float()
       # ====================================================================
		# 3. Handle traning labels vs testing structure
		item = {"peaks": torch.from_numpy(binned_peaks)}

		if not self.is_test:
			# pass the raw SMILES string (your decoder tokenizer will covert this to integrers later)
			item["smiles"] = str(row["normalized_smiles"])
			
		return item

class SpectralCollate:
	def __init__(self, tokenizer ,max_peaks: int = 10000, max_smiles_len: int = 121):
		self.max_peaks = max_peaks
		self.tokenizer = tokenizer
		self.max_smiles_len = max_smiles_len

	def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""
		Dynamically pads the batch to the maximum peak length present IN THE BATCH.
		"""

		# Find the maximum peak Length in the current batch, constrained by max_peaks
		batch_max_len = max(item["peaks"].size(0) for item in batch)
		target_len = min(batch_max_len, self.max_peaks)
		#target_len = batch_max_len
		batch_size = len(batch)

		# Initialize padded tensors and boolean masks
		# Features shape: (batch_size, target_len, 2)
		padded_peaks = torch.zeros(batch_size, target_len, 2, dtype=torch.float32)
		masks = torch.zeros(batch_size, target_len, dtype=torch.bool) # Mask shape: (batch_size, target_len) -> True for real data, False for paddin

		smiles_labels = []
		has_smiles = "smiles" in batch[0]
		tokenized_smiles_list = []

		for i, item in enumerate(batch):
			raw_peaks = item["peaks"]
			actual_len = raw_peaks.size(0)

			# Truncate if individual sample exceeds max_peaks
			slice_len = min(actual_len, target_len)

			# copy over active data points
			padded_peaks[i, :slice_len, :] = raw_peaks[:slice_len, :]
			masks[i, :slice_len] = True

			# 2. Handle Text Tensors (New Tokenization Integration)
			if has_smiles:
				token_ids = self.tokenizer.encode(
					item["smiles"], 
					max_length=self.max_smiles_len, 
					padding="max_length",
					truncation = True,
					return_tensors="pt"
					).squeeze(0) # Removes batch dimension added by encode
				
				tokenized_smiles_list.append(token_ids)
				smiles_labels.append(item["smiles"])

		output = {
			"peaks": padded_peaks,
			"mask":  masks
		}

		if has_smiles:
			output["smiles_token"] = torch.stack(tokenized_smiles_list, dim=0)
			output["smiles"] = smiles_labels 

		return output

		
