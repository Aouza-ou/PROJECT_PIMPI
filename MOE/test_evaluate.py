import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from MOE.Encoder.embadding.adaptive_encoder import AdaptiveSpectralEmbedding
from MOE.Encoder.embadding.dataset import SpectralDataset, SpectralCollate
from MOE.Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer
from MOE.evaluate import run_validation_evaluation

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing standalone evaluation on: {DEVICE}")

    MAX_PEAKS = 2000
    MAX_SMILES_LEN = 182

    # 1. Setup Tokenizer dictionary matching training
    train_parquet = r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet"
    train_df = pd.read_parquet(train_parquet, columns=["normalized_smiles"])
    tokenizer = AtomLevelSmilesTokenizer()
    tokenizer.fit_on_dataframe(train_df["normalized_smiles"])
    vocab_size = tokenizer.vocab_size
    del train_df

    # 2. Initialize Model Architecture Assemblies
    encoder = AdaptiveSpectralEmbedding(max_peaks=MAX_PEAKS, embed_dim=512, latent_dim=512).to(DEVICE)
    decoder = StructuralSmilesDecoder(vocab_size=vocab_size, embed_dim=512, num_layers=4, num_heads=8).to(DEVICE)

    # 3. Load Trained Checkpoint Weights
    checkpoint_path = r"D:\PROJECT_PIMPI\checkpoints\PIMPI_2000.pt"
    print(f"Loading weights from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        print("Successfully loaded model checkpoint weights!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load checkpoint ({e}). Running with untrained weights.")

    # 4. Initialize Validation Dataloader (using a small slice of 10 samples for quick visual check)
    val_parquet = r"D:\PROJECT_PIMPI\MOE\DATA\val_fixed.parquet"
    val_dataset = SpectralDataset(parquet_path=val_parquet, is_test=False)
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(10)))
    
    collate_fn = SpectralCollate(tokenizer=tokenizer, max_peaks=MAX_PEAKS, max_smiles_len=MAX_SMILES_LEN)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # 5. Execute Evaluation & Print Comparisons
    run_validation_evaluation(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        val_loader=val_loader,
        device=DEVICE
    )

if __name__ == "__main__":
    main()