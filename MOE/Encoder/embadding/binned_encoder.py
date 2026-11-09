from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from embadding.positinoal_encoder import PositionalEncoding
from embadding.activation_function import _get_activation

class SpectralVectorEmbadding(nn.Module):

    def __init__(
            self, 
            input_dim: int = 1000, 
            embed_dim: int = 512,
            hidden_dim: int = 1024,
            num_layers: int = 2,
            dropout: float = 0.1,
            use_positional_encoding: bool  = False,
            activation: str = "gelu",
            metadata_dim: int = 4,
            ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding

        layers = []
        current_dim = input_dim

        # Loop handles only hidden layers
        for _ in range(num_layers -1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    _get_activation(activation),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout)
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))

        self.projection = nn.Sequential(*layers)

        if use_positional_encoding:
                self.pos_encoder = PositionalEncoding(embed_dim,dropout)

        self.metadata_projection = nn.Sequential(
                nn.Linear(metadata_dim, hidden_dim // 4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 4),
            )

        self.combine = nn.Sequential(
                nn.Linear(embed_dim + hidden_dim // 4, embed_dim),
                nn.LayerNorm(embed_dim)
            )

        self._init_weights()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SpectralVectorEmbadding":
        """Instantiate model directly from YAML config dictionary."""
        return cls(
            input_dim=cfg["input_dim"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            use_positional_encoding=cfg.get("use_positional_encoding", False),
            activation=cfg.get("activation", "gelu"),
            metadata_dim=cfg.get("metadata", {}).get("input_dim", 4),
        )

    def _init_weights(self):
            for module in self.medules():
                 if isinstance(module, nn.Linear):
                      nn.init.xavier_uniform_(module.weight)
                      if module.bias is not None:
                           nn.init.zeros_(module.bias)

    def forward(
              self, spectra: torch.Tensor, metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
         x = self.projection(spectra)

         if self.use_positional_encoding:
              x = self.pos_encoder(x)
         if metadata is not None:
              meata_embed = self.metadata_projection(metadata)
              x = self.combine(torch.cat([x, meata_embed], dim=1))
         return x



     
        

    