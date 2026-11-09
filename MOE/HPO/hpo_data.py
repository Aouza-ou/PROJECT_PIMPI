import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# 1. FIXED IMPORTS: Use absolute paths matching your training script
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer
from MOE.Encoder.embadding.dataset import SpectralCollate, SpectralDataset
from MOE.MS_configration.config_loader import config

def prepare_tokenizer_and_subset():
    """Fits tokenizer on FULL data, prepares for fast HPO."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fit Tokenizer on FULL data (CPU task, do this once)
    if not config.tokenizer_path.exists():
        print("Fitting tokenizer on full dataset...")
        raw_df = pd.read_parquet(config.full_data_path, columns=["normalized_smiles"])
        tokenizer = AtomLevelSmilesTokenizer()
        tokenizer.fit_on_dataframe(raw_df["normalized_smiles"])
        torch.save({"stoi": tokenizer.stoi, "itos": tokenizer.itos, "vocab_size": tokenizer.vocab_size}, config.tokenizer_path)
        del raw_df
    else:
        tok_data = torch.load(config.tokenizer_path, weights_only=False)
        tokenizer = AtomLevelSmilesTokenizer()
        tokenizer.stoi, tokenizer.itos, tokenizer.vocab_size = tok_data["stoi"], tok_data["itos"], tok_data["vocab_size"]
        
    return tokenizer

def get_hpo_dataloaders(tokenizer, batch_size, max_peaks, subset_size=None):
    """Creates DataLoaders using the fast subset from config.yaml."""
    dataset = SpectralDataset(
        parquet_path=config.subset_data_path, # Pulls from config.yaml paths.subset_filename
        is_test=False
    )

    # --- SMOKE BLOCK ---
    if subset_size is not None and subset_size < len(dataset):
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"*** SMOKE TEST: Using only {len(dataset)} samples ***")

        """ Remove Lettar"""
# -----------------------------------------------------------------------------------

    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Pull max_smiles_len from config.yaml
    max_smiles_len = config.max_smiles_len #['max_smiles_len'] 
    collate_fn = SpectralCollate(tokenizer=tokenizer, max_peaks=max_peaks, max_smiles_len=max_smiles_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    return train_loader, val_loader