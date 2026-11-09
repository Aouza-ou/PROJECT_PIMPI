import sys
from pathlib import Path

# Add project root (D:\PROJECT_PIMPI) to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from MOE.Encoder.embadding.adaptive_encoder import AdaptiveSpectralEmbedding
from MOE.Encoder.embadding.dataset import SpectralDataset, SpectralCollate
from Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer
from evaluate import generate_smiles_autoregressive  # Reusing your generative loop!

def generate_submission_file():
    # 1. Setup Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_PEAKS = 2000
    MAX_SMILES_LEN = 256
    
    print(f"🚀 Initializing Submission Pipeline on: {DEVICE}")

    # 2. Recompile Tokenizer Dictionary from Training Column
    # (Ensures index IDs map identically to how the model was trained)
    train_df = pd.read_parquet("train.parquet", columns=["normalized_smiles"])
    tokenizer = AtomLevelSmilesTokenizer()
    tokenizer.fit_on_dataframe(train_df["normalized_smiles"])
    vocab_size = tokenizer.vocab_size
    del train_df

    # 3. Initialize Model Architecture Assemblies
    encoder = AdaptiveSpectralEmbedding(max_peaks=MAX_PEAKS, embed_dim=512, latent_dim=128).to(DEVICE)
    decoder = StructuralSmilesDecoder(vocab_size=vocab_size, embed_dim=512).to(DEVICE)

    # 4. Load Your Trained Weights
    print("💾 Loading trained model checkpoint weights...")
    # Replace these paths with your actual saved .pt weight files later!
    checkpoint = torch.load("checkpoint/PIMPI_500.pt", map_location=DEVICE)
    encoder.load_state_dict(checkpoint["Encoder parameters"])
    decoder.load_state_dict(checkpoint["Decoder parameters"])
    
    encoder.eval()
    decoder.eval()

    # 5. Initialize Test Dataloader (is_test=True skips looking for answers)
    test_dataset = SpectralDataset(parquet_path="test.parquet", is_test=True)
    collate_fn = SpectralCollate(tokenizer=tokenizer, max_peaks=MAX_PEAKS, max_silmes_len=MAX_SMILES_LEN)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 6. Run Inference Loop
    print("🧪 Generating chemical SMILES strings for test data...")
    all_predicted_smiles = []
    
    for batch in tqdm(test_loader, desc="Predicting"):
        peaks = batch["peaks"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        
        predictions = generate_smiles_autoregressive(
            encoder=encoder, decoder=decoder, tokenizer=tokenizer,
            peaks=peaks, mask=mask, device=DEVICE
        )
        all_predicted_smiles.extend(predictions)

    # 7. Map Predictions directly into the Sample Submission Format
    print("📊 Writing final submission spreadsheet file...")
    submission_df = pd.read_csv("sample_submission.csv")
    
    # Update the target column name to match your challenge specification
    submission_df["normalized_smiles"] = all_predicted_smiles
    
    submission_df.to_csv("submission.csv", index=False)
    print("🎉 Success! Your finalized 'submission.csv' is ready for deployment.")

if __name__ == "__main__":
    generate_submission_file()
