import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

class SpectralDataset(Dataset):
    def __init__(self, parquet_path: str, is_test: bool = False):
        """
        Loads the Parquet file into memory and serves raw arrays.
        """
        # Load the parquet file efficiently using pyarrow
        self.df = pd.read_parquet(parquet_path, engine="pyarrow")
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 1. Extract raw lists/arrays from the row
        mzs = np.array(row["ms2_mzs"], dtype=np.float32)
        intensities = np.array(row["ms2_normalized_intensities"], dtype=np.float32)
        
        # 2. Stack them side-by-side -> Shape: (num_peaks, 2)
        peaks = np.stack([mzs, intensities], axis=-1)
        peaks_tensor = torch.tensor(peaks, dtype=torch.float32)
        
        # 3. Handle training labels vs testing structure
        item = {"peaks": peaks_tensor}
        
        if not self.is_test:
            # Pass the raw SMILES string (your decoder tokenizer will convert this to integers later)
            item["smiles"] = str(row["normalized_smiles"])
            
        return item


class SpectralCollate:
    def __init__(self, max_peaks: int = 500):
        self.max_peaks = max_peaks

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dynamically pads the batch to the maximum peak length present IN THIS BATCH.
        """
        # Find the maximum peak length in the current batch, constrained by max_peaks
        batch_max_len = max(item["peaks"].size(0) for item in batch)
        target_len = min(batch_max_len, self.max_peaks)
        
        batch_size = len(batch)
        
        # Initialize padded tensors and boolean masks
        # Features shape: (batch_size, target_len, 2)
        padded_peaks = torch.zeros(batch_size, target_len, 2, dtype=torch.float32)
        # Mask shape: (batch_size, target_len) -> True for real data, False for padding
        masks = torch.zeros(batch_size, target_len, dtype=torch.bool)
        
        smiles_labels = []
        has_smiles = "smiles" in batch[0]
        
        for i, item in enumerate(batch):
            raw_peaks = item["peaks"]
            actual_len = raw_peaks.size(0)
            
            # Truncate if individual sample exceeds global limit
            slice_len = min(actual_len, target_len)
            
            # Copy over active data points
            padded_peaks[i, :slice_len, :] = raw_peaks[:slice_len, :]
            masks[i, :slice_len] = True
            
            if has_smiles:
                smiles_labels.append(item["smiles"])
                
        output_batch = {
            "peaks": padded_peaks,
            "mask": masks
        }
        
        if has_smiles:
            output_batch["smiles"] = smiles_labels
            
        return output_batch
