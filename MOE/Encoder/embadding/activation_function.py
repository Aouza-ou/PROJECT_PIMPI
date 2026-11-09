import torch.nn as nn

class activation(nn.Module):

	def _get_activation(self, activation: str) -> nn.Module:
		activation = {
			'relu': nn.ReLU(),
			"gelu": nn.GELU(),
			"silu": nn.SiLU(),
			"tanh": nn.Tanh(),

		}
		key = str(activation).lower()

		return activation.get(key, nn.GELU())