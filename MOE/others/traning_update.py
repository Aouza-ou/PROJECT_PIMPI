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

# Make project root importable regardless of where the script is run from
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Data location (single source of truth — change only here)
Path_data = BASE_DIR / "MOE" / "DATA" / "train.parquet"

# Imports matching the modular folder layout
from MOE.Encoder.embadding.adaptive_encoder import AdaptiveSpectralEmbedding
from MOE.Encoder.embadding.dataset import SpectralCollate, SpectralDataset
from MOE.Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer
from tensorboard_helper import init_tensorboard


def train_pipeline():
    # Hyperparameters & Configurations
    BATCH_SIZE = 10
    VAL_BATCH_SIZE = 20
    EPOCHS = 15
    LEARNING_RATE = 5e-4
    MAX_PEAKS = 2000
    MAX_SMILES_LEN = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Folder where all saved models land
    CHECKPOINT_DIR = Path("checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("bertviz_snapshots", exist_ok=True)

    # 1. Fit Tokenizer on the training dataset first
    print("Compiling chemical dictionary from Parquet data table...")
    raw_df = pd.read_parquet(Path_data, columns=["normalized_smiles"])
    tokenizer = AtomLevelSmilesTokenizer()
    tokenizer.fit_on_dataframe(raw_df["normalized_smiles"])
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    del raw_df  # free memory

    # 1b. SAVE THE TOKENIZER — without it you cannot decode predictions later
    torch.save(
        {"stoi": tokenizer.stoi, "itos": tokenizer.itos, "vocab_size": tokenizer.vocab_size},
        CHECKPOINT_DIR / "tokenizer.pt",
    )
    print(f"--> Tokenizer saved: {CHECKPOINT_DIR / 'tokenizer.pt'}")

    # 2. Dataloaders (tokenizer injected into the collate workflow)
    print("Loading data table and preparing train/validation split...")
    dataset = SpectralDataset(parquet_path=str(Path_data), is_test=False)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collate_fn = SpectralCollate(
        tokenizer=tokenizer, max_peaks=MAX_PEAKS, max_smiles_len=MAX_SMILES_LEN
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    print(f"Dataset split complete! Training samples: {train_size} | Validation samples: {val_size}")

    # 3. Initialize complete architecture
    encoder = AdaptiveSpectralEmbedding(
        max_peaks=MAX_PEAKS,
        embed_dim=512,
        latent_dim=512,
    ).to(DEVICE)

    decoder = StructuralSmilesDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=4,
        num_heads=8,
    ).to(DEVICE)

    # TensorBoard logging
    writer = init_tensorboard(
        log_dir="runs/spectral_smiles_experiment",
        encoder=encoder,
        decoder=decoder,
        device=DEVICE,
    )
    print(f"TensorBoard activated. Using hardware accelerator: {DEVICE}")
    print(f"Log directory: {writer.log_dir}")

    # Single optimizer tracking both components
    all_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_parameters, lr=LEARNING_RATE, weight_decay=1e-4)

    # Loss function: Cross-Entropy, ignoring the [PAD] padding positions
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Track the best validation loss for best-model saving
    best_val_loss = float("inf")

    global_step = 0

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"EPOCH {epoch+1}/{EPOCHS}"
        )
        for batch_idx, batch in progress_bar:
            peaks = batch["peaks"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            smiles_token = batch["smiles_token"].to(DEVICE)

            optimizer.zero_grad()

            # A. Encoder step: extract fingerprint (+ attention weights)
            fingerprint, attention_maps = encoder(peaks, mask=mask)

            # B. Causal sequence preparation
            decoder_input = smiles_token[:, :-1]   # [SOS] ... up to penultimate token
            decoder_target = smiles_token[:, 1:]   # first token ... [EOS]

            # C. Causal mask
            seq_len = decoder_input.size(1)
            tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.float32)

            # D. Decoder step: predicted logits
            logits = decoder(
                tgt_tokens=decoder_input,
                memory_fingerprint=fingerprint,
                tgt_mask=tgt_mask,
            )

            # E. Loss: reshape for CrossEntropy (N, C) vs (N)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                decoder_target.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if batch_idx == len(train_loader) - 1:
                snapshot_path = f"bertviz_snapshots/attention_epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "attention": attention_maps.detach().cpu(),
                        "peaks": peaks.detach().cpu(),
                        "smiles": batch.get("smiles", None),
                    },
                    snapshot_path,
                )
                print(f"--> Saved BertViz matrix snapshot to: {snapshot_path}")

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch + 1)
        print(f"=== Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f} ===")

        # ==================== VALIDATION ====================
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"VALIDATING {epoch+1}/{EPOCHS}")

        with torch.no_grad():
            for val_batch in val_bar:
                peaks = val_batch["peaks"].to(DEVICE)
                mask = val_batch["mask"].to(DEVICE)
                smiles_token = val_batch["smiles_token"].to(DEVICE)

                fingerprint, _ = encoder(peaks, mask=mask)
                decoder_input = smiles_token[:, :-1]
                decoder_target = smiles_token[:, 1:]

                seq_len = decoder_input.size(1)
                tgt_mask = decoder.generate_causal_mask(seq_len=seq_len, device=DEVICE, dtype=torch.float32)

                logits = decoder(
                    tgt_tokens=decoder_input,
                    memory_fingerprint=fingerprint,
                    tgt_mask=tgt_mask,
                )

                loss = criterion(logits.reshape(-1, vocab_size), decoder_target.reshape(-1))
                val_loss += loss.item()
                val_bar.set_postfix({"Val_Loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch + 1)
        print(f"--> Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}")

        # ==================== MODEL SAVING ====================

        # 1) Save a checkpoint EVERY epoch (full training state, resumable)
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
        epoch_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"--> Checkpoint saved: {epoch_path}")

        # 2) Save the BEST model so far (weights only — for inference)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = CHECKPOINT_DIR / "best_model.pt"
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
            print(f"--> New best model saved (val_loss={avg_val_loss:.4f}): {best_path}")

        # Back to training mode for the next epoch
        encoder.train()
        decoder.train()

    writer.close()
    print("Complete Encoder-Decoder Training Cycle Finished Successfully.")


if __name__ == "__main__":
    train_pipeline()