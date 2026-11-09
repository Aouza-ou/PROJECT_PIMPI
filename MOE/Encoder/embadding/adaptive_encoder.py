import torch
from typing import Optional, Dict, Any
import torch.nn as nn
# Assuming your other classes are imported or in the same file
from .activation_function import activation
from .positional_encoder import PositionalEncoding
from MOE.FeedForwardNetwork.FFN import  FeedForwardNeuralNetwork as FFn

class AdaptiveSpectralEmbedding(nn.Module):
       def __init__(
        self,
        max_peaks: int,
        embed_dim: int,
        latent_dim: int,
        num_heads: int,       # Added for the MultiheadAttention
        peak_hidden_dim: int, # Added for the nn.Linear(2, 64)
        dropout: float,
        act_type: str
    ):
        super().__init__()

        self.max_peaks = max_peaks
        self.embed_dim = embed_dim

        # Dynamically fetch the activation function from your custom class 
        act_lookup = activation() 
        chosen_act = act_lookup._get_activation(act_type)

    # ====================================================================================
        # Peak embedding (m/z + intensity) using the chosen activation (MLP Vecort embedding)
        self.peak_encoder = nn.Sequential(
            nn.Linear(2, peak_hidden_dim), 
            chosen_act, # Used your dynamic activation selection here!
            nn.Linear(peak_hidden_dim, embed_dim)     
        )
    # ====================================================================================
        # Added Positional Encoding Module here!
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout, max_len=max_peaks)

    # ====================================================================================
        # Attention-based aggregation (Transformer [MHA self-attention]): GPT
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    # ====================================================================================
        # Layer normalization (ADD & NORM)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # ====================================================================================
        # FeedForward Pass Neural Network for final embedding refinement (brain of the model)
        self.FFN = FFn(input_dim=embed_dim, # input dimension is the embedding dimension after pooling.
                       hidden_dim=embed_dim*2, # hidden dimension is double the embedding dimension for richer representation.
                       output_dim=embed_dim,  # output dimension is the same as embedding dimension for final fingerprint (shrinked representation).
                       dropout=dropout) # dropout for regularization.
 
    # ====================================================================================
    # Forward Block: the main logic of the Adaptive Spectral Embedding Model.
       def forward(
         self, 
         peaks: torch.Tensor,
         mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Encode individual peaks
        x = self.peak_encoder(peaks) # (batch_size, num_peaks, embed_dim)
        
        # 2. Inject Positional Information before the Attention step!
        x = self.positional_encoding(x)


        padding_mask = None
        if mask is not None:
            padding_mask = ~mask if mask.dtype == torch.bool else (mask == 0)
            all_mask = padding_mask.all(dim=1)
            if all_mask.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_mask, 0] = False

        # Self-attention ( Q = K = V = x ) with padding mask
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=padding_mask)
            
        # Residual connection + Layer Normalization (ADD & NORM)
        x = self.norm(x + attn_out)

        # Global pooling (mean of valid peaks only)
        if mask is not None:
            valid_mask = mask.unsqueeze(-1).float() if mask.dtype == torch.bool else (mask != 0).unsqueeze(-1).float()
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        # Apply dropout to pooled vector
        x = self.dropout(x)

        # Final refinement through the FeedForward Neural Network.
        ffn_out = self.FFN(x) 
        return ffn_out, attn_weights
