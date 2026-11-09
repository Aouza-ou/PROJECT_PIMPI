import torch.nn as nn
import torch 
from MOE.Encoder.embadding.activation_function import activation



class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,output_dim:int, dropout: float = 0.1,):
        """
        Args:
            input_dim: The dimension of the vector right after Global Pooling.
            hidden_dim: Internal capacity for refinement.
            output_dim: The final size of your single, clean structural fingerprint.
        """
        super().__init__()
        activation_function = activation()._get_activation("gelu" ) # activation function for. non-linearty.
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # maps to internal capacity.
            activation_function, # non-linearity.
            nn.Dropout(dropout), # droput layer to prevent overfitting
            nn.Linear(hidden_dim, output_dim) # maps to final Fingerprint size.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) # x shape expected: (batch_size, input_dim) after global-pooling.
        
   



