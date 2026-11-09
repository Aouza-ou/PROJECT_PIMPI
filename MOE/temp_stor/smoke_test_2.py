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
#sys.path.append(os.path.abspath("D:\\PROJECT_PIMPI"))
#sys.path.append(os.path.abspath("D:\\PROJECT_PIMPI\\MOE"))
# Position: training.py (Lines 9–10 and 31)
# Replace hardcoded sys.path appends and Path_data with:
BASE_DIR = Path(__file__).resolve().parents[1]
Path_data = BASE_DIR / "MOE" / "DATA" / "train.parquet"
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
    VAL_BATCH_SIZE =10
    EPOCHS = 20
    LEARNING_RATE = 5e-4
    MAX_PEAKS = 2000
    MAX_SMILES_LEN = 182
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== SMOKE-TEST MODE =============================
    SMOKE_TEST = True
    SMOKE_SAMPLES = 2000
    if SMOKE_TEST:
        EPOCHS = EPOCHS
        LEARNING_RATE = LEARNING_RATE
        VALDATA = 200
    


    #Folder where all saved models land 
    CHECKPOINT_DIR = Path("checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("bertviz_snapshots", exist_ok=True)

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

    # Save the tokenizer - without it you cannot decode predictions leter.
    torch.save(
        {"stoi": tokenizer.stoi, "itos": tokenizer.itos, "vocab_size":
         tokenizer.vocab_size
         },
         CHECKPOINT_DIR / "tokenizer.pt",
    )
    print(f"--> Tokenizer saved: {CHECKPOINT_DIR / 'tokenizer.pt'}")


    # 3. FIXED DATALOADER: Inject the initialized tokenizer into the collate workflow
    print("Loading data table and preparing train/validation split...")
    dataset = SpectralDataset(parquet_path=Path_data, is_test=False)

    # --- Smoke test: shirnk the dataset to a small random subset ---
    if SMOKE_TEST:
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset))[:SMOKE_SAMPLES].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"*** SMOKE TEST MODE: USING ONLY {len(dataset)} SAMPLES ***")

    # the subset IS the training set
    train_dataset = dataset
    

    # Excute the virtual split with a fixed random seed for consistent results
    val_dataset = SpectralDataset(
        parquet_path=str(r"D:\PROJECT_PIMPI\MOE\DATA\val_fixed.parquet"),
        is_test = False
    )
    # Validate on a fixed 2000-row slice of the exam (fast, stable, same every run)
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(VALDATA)))  

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

    print(f"Dataset split complete! Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
   
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

     # Initialize TensorBoard Logging
    writer = init_tensorboard(
        log_dir=f"runs/PIMPI_{len(train_dataset)}",
        encoder=encoder,
        decoder=decoder,
        device=DEVICE
        )
    print(f"TensorBoard activated. Using hardware accelerator: {DEVICE}")
    print(f"Log directory: {writer.log_dir}") # verify where logs actually land.

    # Gather parameters from both components for a single optimizer tracking system
    all_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_parameters, lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 5. FIXED LOSS FUNCTION: Use Cross-Entropy and ignore the [PAD] padding values
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

     # Track the best validation loss for best-model saving
    best_val_loss = float("inf")

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
            tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.bool)

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
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                #writer.flush()
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
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, global_step)
        print(f"=== Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f} ===")

        # ==================== VALIDATION ====================
         # Free GPU memory left over from training before validation
        del fingerprint, attention_maps, logits, loss
        torch.cuda.empty_cache()

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'VALIDATING {epoch+1}/{EPOCHS}')

        with torch.no_grad():
            for val_batch in val_bar:
                peaks = val_batch["peaks"].to(DEVICE)
                mask = val_batch["mask"].to(DEVICE)
                smiles_token = val_batch["smiles_token"].to(DEVICE)
                # Forward pass only
                fingerprint, _ = encoder(peaks, mask=mask)
                decoder_input = smiles_token[:, :-1]
                decoder_target = smiles_token[:, 1:]

                seq_len = decoder_input.size(1)
                tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.bool)

                logits = decoder(
                    tgt_tokens=decoder_input,
                    memory_fingerprint = fingerprint,
                    tgt_mask=tgt_mask
                )

                loss = criterion(logits.reshape(-1, vocab_size), decoder_target.reshape(-1))
                val_loss += loss.item() # Accumulate loss for each batch
                val_bar.set_postfix({"Val_Loss": f"{loss.item():.4f}"})
        
        # Log mid-epoch validation loss to TensorBoard using the exact same global_step
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch+1)
        # writer.flush()
        print(f"--> Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}")

        # ============== MODEL SAVING =========================

        # Save a checkpoint every epoch full traning sate,  resumable
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
        }
        epoch_path = CHECKPOINT_DIR / f"PIMPI_{len(train_dataset)}_EPOCH_{epoch+1}.pt"
        best_path = CHECKPOINT_DIR / f"PIMPI_{len(train_dataset)}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"--> Checkpoint saved: {epoch_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
           # best_path = CHECKPOINT_DIR / "PIMPI.pt"
            torch.save(
                 {
                   "epoch": epoch + 1,
                   "encoder_state_dict": encoder.state_dict(),
                   "decoder_state_dict": decoder.state_dict(),
                   "tokenizer_vocab": {"stoi": tokenizer.stoi, "itos": tokenizer.itos},
                   "vocab_size": vocab_size,
                   "val_loss": avg_val_loss,
                },
                best_path,
            )
            print(f"--> .New best model saved (val_loss={avg_val_loss:.4f}), {best_path}")
         # Switch back to training mode immediately
        encoder.train()
        decoder.train()

    writer.close()
    print("Complete Encoder-Decoder Training Cycle Finished Successfully.")
# tensorboard --logdir=runs

if __name__ == "__main__":
    train_pipeline()
