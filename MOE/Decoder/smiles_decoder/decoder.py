import torch
import torch.nn as nn
from typing import Optional
from MOE.Encoder.embadding.activation_function import activation
# Add this to your imports at the top of decoder.py
from MOE.MS_configration.config_loader import config
 # Adjust 'MOE.' if your folder structure is different

class StructuralSmilesDecoder(nn.Module):
    def __init__(
    self,
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    act_type: str
):
        super().__init__()
        # ... (all your existing initialization code stays identical here) ...
        self.token_embedding= nn.Embedding(vocab_size, embed_dim)
        max_len = config.decoder_config["max_seq_len"]  # Fetch max sequence length from config
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # --- Memory expansion (the fix) ---
        #self.num_memory_tokens = 8
        self.num_memory_tokens = config.decoder_config['num_memory_tokens']
         # Dynamically fetch the activation function from your custom class 
        act_lookup = activation() 
        chosen_act = act_lookup._get_activation(act_type)
        self.memory_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            chosen_act,
            nn.Linear(embed_dim, embed_dim)
        )
        self.memory_codes = nn.Parameter(torch.randn(1, self.num_memory_tokens, embed_dim)*0.02)
        self.dropout = nn.Dropout(dropout)

    # ====================================================================================
    # FIXED: Encapsulated cleanly as a Static Method directly inside the decoder class
    # ====================================================================================
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Generates a 2D causal mask to prevent the decoder from looking ahead.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        causal_mask = causal_mask.masked_fill(mask, float('-inf'))
        return causal_mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory_fingerprint: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = tgt_tokens.size()
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(seq_len, device=tgt_tokens.device)

        # Position: StructuralSmilesDecoder.forward (Lines 36–38)
        if seq_len > self.pos_embedding.size(1):
           raise ValueError(f"Sequence length ({seq_len}) exceeds decoder max positional limit ({self.pos_embedding.size(1)}).")

        x = self.token_embedding(tgt_tokens) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        if memory_fingerprint.dim() == 2:
            memory_fingerprint = memory_fingerprint.unsqueeze(1)

            # 1) Refine the fingerprint
        mem = self.memory_proj(memory_fingerprint)
            # 2) Expand: give each of the 8 slots the fingerprint + its own learned code
        memory = mem + self.memory_codes
            
        decoder_out = self.transformer_decoder(
            tgt=x, memory=memory, tgt_mask=tgt_mask
        )
        
        return self.lm_head(decoder_out)
