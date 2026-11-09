import sys
from pathlib import Path

# Add project root (D:\PROJECT_PIMPI) to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm

from MOE.Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.Tokennizer.smiles_tokenizer import AtomLevelSmilesTokenizer

@torch.no_grad()
def generate_smiles_autoregressive(
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: AtomLevelSmilesTokenizer,
    peaks: torch.Tensor,
    mask: torch.Tensor,
    max_length: int = 182,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """
    Takes raw spectrum matrices, runs the encoder, and uses an autoregressive
    greedy decoding loop to generate final human-readable SMILES chemical strings.
    """
    encoder.eval()
    decoder.eval()
    
    batch_size = peaks.size(0)
    
    # 1. ENCODER PASS: Generate the structural fingerprint blueprint
    # Catch both outputs but discard attention weights (_) for inference optimization
    fingerprint, _ = encoder(peaks.to(device), mask=mask.to(device))
    
    # 2. INITIALIZE WORKSPACE: Set up every sample in the batch with a [SOS] token
    sos_idx = tokenizer.stoi[tokenizer.sos_token]
    eos_idx = tokenizer.stoi[tokenizer.eos_token]
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    
    # Start sequence tensor shape: (batch_size, 1) filled with [SOS]
    generated_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    # Track which items in the batch have hit an [EOS] marker to stop updating them
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # 3. AUTOREGRESSIVE GENERATION LOOP
    for step in range(max_length - 1):
        # Generate the causal attention matrix matching our current token length
        seq_len = generated_tokens.size(1)
        # Call the static mask generator directly via the Decoder class layout
        tgt_mask = StructuralSmilesDecoder.generate_causal_mask(seq_len=seq_len, device=device)
        
        # DECODER PASS: Get probability logits
        logits = decoder(
            tgt_tokens=generated_tokens,
            memory_fingerprint=fingerprint,
            tgt_mask=tgt_mask
        )
        
        # Extract predictions for the VERY LAST character position only -> Shape: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]
        
        # GREEDY SELECTION: Extract the single index with the absolute maximum score
        next_tokens = torch.argmax(next_token_logits, dim=-1) # Shape: (batch_size,)
        
        # If a sample was already finished, force its new tokens to remain [PAD]
        next_tokens = torch.where(finished, torch.tensor(pad_idx, device=device), next_tokens)
        
        # Append the new characters onto our running text matrix -> Shape grows to (batch_size, seq_len + 1)
        generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
        
        # Update our finished markers if the model generated an [EOS] token
        finished |= (next_tokens == eos_idx)
        
        # Break early if every single molecule in the batch has completed its string
        if finished.all():
            break
            
    # 4. DECODING STRINGS: Translate integer matrices back into standard text
    predicted_smiles = []
    for i in range(batch_size):
        raw_token_ids = generated_tokens[i]
        text_smiles = tokenizer.decode(raw_token_ids)
        predicted_smiles.append(text_smiles)
        
    return predicted_smiles

@torch.no_grad()
def run_validation_evaluation(encoder, decoder, tokenizer, val_loader, device):
    """
    Streams your unseen validation data loader, generates SMILES strings,
    and prints direct text comparisons to monitor true chemical learning.
    """
    print("\n****Running Validation Chemical Generation Pass...")
    all_predictions = []
    all_targets = []
    
    # Process validation data stream using our clean progress bars
    for batch in tqdm(val_loader, desc="Loading Weight"):
        peaks = batch["peaks"].to(device)
        mask = batch["mask"].to(device)
        
        # Call our autoregressive engine
        predictions = generate_smiles_autoregressive(
            encoder=encoder, decoder=decoder, tokenizer=tokenizer,
            peaks=peaks, mask=mask, device=device
        )
        
        all_predictions.extend(predictions)
        # If your collate function logs text labels, retrieve them to check accuracy
        if "smiles" in batch:
            all_targets.extend(batch["smiles"])
        elif "smiles_token" in batch:
            # Decode gold standard tokens back into text for comparison
            for token_seq in batch["smiles_token"]:
                all_targets.append(tokenizer.decode(token_seq))
            
    # Print out a quick snippet of text comparisons so you can inspect them visually
    print("\n📝 --- CHEMICAL PREDICTION SAMPLES ---")
    num_samples_to_show = min(10, len(all_predictions))
    for i in range(num_samples_to_show):
        print(f"\n[Sample {i+1}]")
        if all_targets and i < len(all_targets):
            print(f"   Target Answer: {all_targets[i]}")
        print(f"   Model Output : {all_predictions[i]}")
    print("---------------------------------------")
    
    return all_predictions
