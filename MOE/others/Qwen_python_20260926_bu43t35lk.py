"""
Vector Embedding Module for MS/MS Spectra
==========================================
Plug-and-play module that converts raw spectral features into dense vector embeddings.

Input: Spectral features (m/z bins, intensities, metadata)
Output: Fixed-size embedding vector

This module is designed to be:
- Modular: Can be swapped with other embedding strategies
- Configurable: Different architectures via config
- Testable: Works standalone with mock data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import math


class SpectralVectorEmbedding(nn.Module):
    """
    Converts MS/MS spectral data into dense vector embeddings.
    
    Architecture:
    - Input projection: Maps raw spectral features to hidden dimension
    - Positional encoding: Adds spectral position information
    - Layer normalization: Stabilizes training
    - Optional dropout: For regularization
    
    Args:
        input_dim (int): Dimension of input spectral features
        embed_dim (int): Dimension of output embedding
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of projection layers
        dropout (float): Dropout rate
        use_positional_encoding (bool): Whether to add positional encoding
        max_seq_len (int): Maximum sequence length for positional encoding
    """
    
    def __init__(
        self,
        input_dim: int = 1000,  # Default: 1000 m/z bins
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_seq_len: int = 1000,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Build projection layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Final projection to embed_dim
        layers.append(nn.Linear(current_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Positional encoding (optional)
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_len)
        
        # Metadata embedding (precursor m/z, collision energy, etc.)
        self.metadata_projection = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # precursor_mz, charge, CE, ionization_mode
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # Final combination layer
        self.combine = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.GELU())
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        spectra: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            spectra: Tensor of shape (batch_size, input_dim)
                    Contains binned spectral intensities
            metadata: Optional tensor of shape (batch_size, num_metadata_features)
                     Contains precursor_mz, charge, collision_energy, ionization_mode
        
        Returns:
            embeddings: Tensor of shape (batch_size, embed_dim)
        """
        # Project spectral features
        x = self.projection(spectra)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Process metadata if provided
        if metadata is not None:
            meta_embed = self.metadata_projection(metadata)
            x = self.combine(torch.cat([x, meta_embed], dim=-1))
        
        return x
    
    def get_output_dim(self) -> int:
        """Return the output embedding dimension"""
        return self.embed_dim


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for spectral embeddings.
    Helps the model understand the position/order of spectral features.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model)
        """
        if x.dim() == 2:
            # If input is (batch, d_model), add sequence dimension
            x = x.unsqueeze(1)
            x = x + self.pe[:, :x.size(1), :]
            x = x.squeeze(1)  # Remove sequence dimension
        else:
            x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class AdaptiveSpectralEmbedding(nn.Module):
    """
    Alternative embedding strategy using adaptive binning.
    Useful when spectra have variable numbers of peaks.
    
    This module demonstrates the plug-and-play nature - 
    you can swap this with SpectralVectorEmbedding.
    """
    
    def __init__(
        self,
        max_peaks: int = 500,
        embed_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_peaks = max_peaks
        self.embed_dim = embed_dim
        
        # Peak embedding (m/z + intensity)
        self.peak_encoder = nn.Sequential(
            nn.Linear(2, 64),  # m/z, intensity
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        
        # Attention-based aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        peaks: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            peaks: (batch_size, num_peaks, 2) - m/z and intensity pairs
            mask: (batch_size, num_peaks) - True for valid peaks
        """
        # Encode individual peaks
        x = self.peak_encoder(peaks)  # (batch, num_peaks, embed_dim)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm(x + attn_out)
        
        # Global pooling (mean of valid peaks)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.dropout(x)


# ============ Testing & Usage Example ============

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SpectralVectorEmbedding Module")
    print("=" * 60)
    
    # Configuration
    config = {
        "input_dim": 1000,      # 1000 m/z bins
        "embed_dim": 512,       # Output embedding size
        "hidden_dim": 1024,     # Hidden layer size
        "num_layers": 2,
        "dropout": 0.1,
        "use_positional_encoding": True
    }
    
    # Initialize module
    embedding = SpectralVectorEmbedding(**config)
    print(f"\n✓ Module initialized with config: {config}")
    print(f"✓ Total parameters: {sum(p.numel() for p in embedding.parameters()):,}")
    
    # Create mock data (simulating batch of spectra)
    batch_size = 32
    mock_spectra = torch.randn(batch_size, config["input_dim"])
    mock_metadata = torch.randn(batch_size, 4)  # precursor_mz, charge, CE, ion_mode
    
    print(f"\n✓ Mock spectra shape: {mock_spectra.shape}")
    print(f"✓ Mock metadata shape: {mock_metadata.shape}")
    
    # Forward pass
    embeddings = embedding(mock_spectra, mock_metadata)
    print(f"\n✓ Output embeddings shape: {embeddings.shape}")
    print(f"✓ Expected shape: ({batch_size}, {config['embed_dim']})")
    
    # Verify output
    assert embeddings.shape == (batch_size, config["embed_dim"]), "Shape mismatch!"
    assert not torch.isnan(embeddings).any(), "NaN values detected!"
    assert not torch.isinf(embeddings).any(), "Inf values detected!"
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Module is ready for integration.")
    print("=" * 60)
    
    # Test adaptive embedding
    print("\n" + "=" * 60)
    print("Testing AdaptiveSpectralEmbedding Module")
    print("=" * 60)
    
    adaptive_emb = AdaptiveSpectralEmbedding(max_peaks=500, embed_dim=512)
    mock_peaks = torch.randn(batch_size, 500, 2)  # 500 peaks, each with m/z and intensity
    
    adaptive_out = adaptive_emb(mock_peaks)
    print(f"✓ Adaptive output shape: {adaptive_out.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in adaptive_emb.parameters()):,}")