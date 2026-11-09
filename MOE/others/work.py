import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(os.path.abspath("D:\\PROJECT_PIMPI"))
sys.path.append(os.path.abspath("D:\\PROJECT_PIMPI\\MOE"))

# 1. FIXED IMPORTS: Correct paths matching modular folder layout

from MOE.Encoder.embadding.adaptive_encoder import AdaptiveSpectralEmbedding
from MOE.Encoder.embadding.dataset import SpectralCollate, SpectralDataset
from MOE.Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer
from tensorboard_helper import init_tensorboard

# ====================== Imports ======================================== #

def train_pipeline():
    # Hyperparameters & Configurations
    BATCH_SIZE = 10
    VAL_BATCH_SIZE = 20
    EPOCHS = 15
    LEARNING_RATE = 5e-4
    MAX_PEAKS = 500
    MAX_SMILES_LEN = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # 2. FIXED: Fit Tokenizer on your training dataset first
    print("Compiling chemical dictionary from Parquet data table...")
    Path_data = r"D:\PROJECT_PIMPI\MOE\DATA\train.parquet"
    raw_df = pd.read_parquet(Path_data, columns=["normalized_smiles"])
    tokenizer = AtomLevelSmilesTokenizer()
    tokenizer.fit_on_dataframe(raw_df["normalized_smiles"])
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    
    # Free up memory from raw pandas reading
    del raw_df

    # 3. FIXED DATALOADER: Inject the initialized tokenizer into the collate workflow
    print("Loading data table and preparing train/validation split...")
    dataset = SpectralDataset(parquet_path=Path_data, is_test=False)

    # Calculate exact subset split integer sizes
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Excute the virtual split with a fixed random seed for consistent results
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size,
        val_size],
        generator=torch.Generator().manual_seed(42)

    )

    collate_fn = SpectralCollate(tokenizer=tokenizer, max_peaks=MAX_PEAKS, max_smiles_len=MAX_SMILES_LEN)
    
    train_loader = DataLoader(
       train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Dataset split complete! Training samples: {train_size} | Validation samples: {val_size}")
   
    # 4. INITIALIZE COMPLETE ARCHITECTURE
    encoder = AdaptiveSpectralEmbedding(
        max_peaks=MAX_PEAKS,
        embed_dim=512,
        latent_dim=512  # Target size for the structural fingerprint bottleneck
    ).to(DEVICE)
    
    decoder = StructuralSmilesDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=4,
        num_heads=8
    ).to(DEVICE)

     # Initialize TensorBoard Logging (one writer -> all metrics, tags decide grouping)
    writer = init_tensorboard(
        log_dir="runs/spectral_smiles_experiment",
        encoder=encoder,
        decoder=decoder,
        device=DEVICE
        )
    print(f"TensorBoard activated. Using hardware accelerator: {DEVICE}")
    print(f"Log directory: {writer.log_dir}")  # ✅ CHANGED: verify where logs actually land

    # Gather parameters from both components for a single optimizer tracking system
    all_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_parameters, lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 5. FIXED LOSS FUNCTION: Use Cross-Entropy and ignore the [PAD] padding values
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    global_step = 0

    for epoch in range(EPOCHS):
        os.makedirs("bertviz_snapshots", exist_ok=True)  #  <- Create a directory to hold snapshots for BertViz inspection
        encoder.train()
        decoder.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'EPOCH {epoch+1}/{EPOCHS}')
        for batch_idx, batch in progress_bar:
            # Extract data matrices from dataloader batch
            peaks = batch["peaks"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            smiles_token = batch["smiles_token"].to(DEVICE) # Shape: (batch_size, max_smiles_len)
            
            optimizer.zero_grad()

            # A. ENCODER STEP: Extract the fingerprint and catch the attention weights matrix
            fingerprint, attention_maps = encoder(peaks, mask=mask)

            # B. TEXT DATA PREPARATION FOR CAUSAL SEQUENCES
            # Decoder Input: drops the last token (e.g. [SOS] to penultimate token)
            decoder_input = smiles_token[:, :-1]
            # Decoder Target: shifted left by one slot (e.g. first token to [EOS])
            decoder_target = smiles_token[:, 1:]
            
            # C. CAUSAL MASK GENERATION: Keep track of target sequence length
            seq_len = decoder_input.size(1)
            tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.float32)

            # D. DECODER STEP: Compute predicted probability grids (logits)
            logits = decoder(
                tgt_tokens=decoder_input, 
                memory_fingerprint=fingerprint, 
                tgt_mask=tgt_mask
            )

            # E. COMPLEX RESHAPING FOR TARGET LOSS ALIGNMENT
            # CrossEntropy expects predictions as shape (N, C) and targets as shape (N)
            loss = criterion(
                logits.reshape(-1, vocab_size), 
                decoder_target.reshape(-1)
            )
            
            # Optimization block
            loss.backward()
            # Gradient clipping to protect model weights from exploding
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # Log step metrics to TensorBoard every 10 steps
            if batch_idx % 10 == 0:
                writer.add_scalar("Train/Step_Loss", loss.item(), global_step)          # ✅ CHANGED tag
                writer.add_scalar("Progress/Current_Epoch", epoch + 1, global_step)
                writer.add_scalar("Progress/Batch_index", batch_idx, global_step)
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if batch_idx == len(train_loader) -1:
                snapshot_path=f"bertviz_snapshots/attention_epoch_{epoch+1}.pt"
                torch.save({
                     "attention": attention_maps.detach().cpu(),
                     "peaks": peaks.detach().cpu(),
                     "smiles": batch.get("smiles", None)
                }, snapshot_path)
                print(f"--> Saved BertViz matrix snapshot to: {snapshot_path}")

        # Summary Metrics per Epoch
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, global_step)                  # ✅ CHANGED tag
        print(f"=== Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f} ===")


        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                peaks = val_batch["peaks"].to(DEVICE)
                mask = val_batch["mask"].to(DEVICE)
                smiles_token = val_batch["smiles_token"].to(DEVICE)
                # Forward pass only
                fingerprint, _ = encoder(peaks, mask=mask)
                decoder_input = smiles_token[:, :-1]
                decoder_target = smiles_token[:, 1:]

                seq_len = decoder_input.size(1)
                tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.float32)

                logits = decoder(
                    tgt_tokens=decoder_input,
                    memory_fingerprint = fingerprint,
                    tgt_mask=tgt_mask
                )

                loss = criterion(logits.reshape(-1, vocab_size), decoder_target.reshape(-1))
                val_loss += loss.item() # Accumulate loss for each batch

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, global_step)                  # ✅ CHANGED: correct tag + exactly 3 args
        print(f"--> Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}")

    writer.close()
    print("Complete Encoder-Decoder Training Cycle Finished Successfully.")
# tensorboard --logdir=runs

if __name__ == "__main__":
    train_pipeline()