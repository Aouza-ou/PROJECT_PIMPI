import torch
from tqdm import tqdm

def run_hpo_epoch(encoder, decoder, train_loader, val_loader, criterion, optimizer, device, pad_idx, vocab_size, all_parameters, max_norm=1.0):
    """Runs exactly 1 epoch of training and validation. Returns val_loss for Optuna pruning."""
    #criterion.ingnore_index = pad_idx
    
    # --- TRAIN ---
    encoder.train()
    decoder.train()
    train_loss = 0.0

    training_pbar = tqdm(train_loader, desc=' Training', leave=False)
    for batch in training_pbar:
        peaks = batch["peaks"].to(device)
        mask = batch["mask"].to(device)
        smiles_token = batch["smiles_token"].to(device)
        
        optimizer.zero_grad()
        fingerprint, _ = encoder(peaks, mask=mask)
        
        decoder_input = smiles_token[:, :-1]
        decoder_target = smiles_token[:, 1:]
        tgt_mask = decoder.generate_causal_mask(seq_len=decoder_input.size(1), device=device, dtype=torch.bool)
        
        logits = decoder(tgt_tokens=decoder_input, memory_fingerprint=fingerprint, tgt_mask=tgt_mask)
        
        # Match training.py: reshape using vocab_size
        loss = criterion(logits.reshape(-1, vocab_size), decoder_target.reshape(-1))
        
        loss.backward()
        # Match training.py: clip all_parameters together
        torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=max_norm)
        optimizer.step()
        train_loss += loss.item()
        training_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # --- VALIDATE ---
    encoder.eval()
    decoder.eval()
    val_loss = 0.0

    val_pbar = tqdm(val_loader, desc=' Validation', leave=False)
    with torch.no_grad():
        for batch in val_pbar:
            peaks = batch["peaks"].to(device)
            mask = batch["mask"].to(device)
            smiles_token = batch["smiles_token"].to(device)
            
            fingerprint, _ = encoder(peaks, mask=mask)
            decoder_input = smiles_token[:, :-1]
            decoder_target = smiles_token[:, 1:]
            tgt_mask = decoder.generate_causal_mask(seq_len=decoder_input.size(1), device=device, dtype=torch.bool)
            
            logits = decoder(tgt_tokens=decoder_input, memory_fingerprint=fingerprint, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, vocab_size), decoder_target.reshape(-1))
            val_loss += loss.item()
            val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return val_loss / len(val_loader)