"""
Positinoal Embadding Module for MS/MS Spectra
=============================================
Plug-and-play module that converts raw spectral features into dens vector embadding.

Input: Spectral features (m/z bins, intensities, metadata)
Output: Fixed-size embedding vector

This module is designed to be:
- Modular: Can be swapped with other embedding strategies
- Configurable: Different architectures via config
- Testable: Works stand
"""
# 0. ML / AI lybarys
import torch
import torch.nn as nn
#import torch.nn.functional 
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for spectral embeddings.
    Helps the model understand the position/order of spectral features.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model)
            Add an explicit validation check before tensor addition to verify that the sequence length does not exceed max_len
        """
        seq_len = x.size(1) if x.dim() != 2 else 1
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)}).")
        
        if x.dim() == 2:
            # If input is (batch, d_model), add sequence dimension
            x = x.unsqueeze(1)
            x = x + self.pe[:, :x.size(1), :]
            x = x.squeeze(1)  # Remove sequence dimension
        else:
            x = x + self.pe[:, :x.size(1), :]

       
        return self.dropout(x)

