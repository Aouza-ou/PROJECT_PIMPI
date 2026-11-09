import torch
import torch.nn as nn
import torch.optim as optim
from MOE.Encoder.embadding.adaptive_encoder import AdaptiveSpectralEmbedding
from MOE.Decoder.smiles_decoder import StructuralSmilesDecoder
from MOE.MS_configration.config_loader import config

def build_model_and_optimizer(hp, vocab_size, pad_idx, device):
    """Instantiates model and optimizer based on Optuna's suggested hyperparameters."""
    
    # Pull non-searched config values from config.yaml
    enc_cfg = config.encoder_config
    dec_cfg = config.decoder_config
    
    encoder = AdaptiveSpectralEmbedding(
        max_peaks=hp["max_peaks"],          # From Optuna HPO
        embed_dim=hp["embed_dim"],          # From Optuna HPO
        latent_dim=hp["latent_dim"],        # From Optuna HPO
        num_heads=hp["num_heads"],          # From Optuna HPO
        peak_hidden_dim=enc_cfg['peak_hidden_dim'], # From config.yaml
        dropout=enc_cfg['dropout'],         # From config.yaml
        act_type=enc_cfg['act_type']        # From config.yaml
    ).to(device)
    
    decoder = StructuralSmilesDecoder(
        vocab_size=vocab_size,
        embed_dim=hp["embed_dim"],          # From Optuna HPO
        num_layers=hp["num_layers"],        # From Optuna HPO
        num_heads=hp["num_heads"],          # From Optuna HPO
        dropout=dec_cfg['dropout'],         # From config.yaml
        act_type=dec_cfg['act_type']        # From config.yaml
    ).to(device)
    
    all_parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(all_parameters, lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    return encoder, decoder, optimizer, criterion, all_parameters