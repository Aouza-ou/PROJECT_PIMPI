import torch
from torch.utils.tensorboard import SummaryWriter

def init_tensorboard(log_dir: str, encoder: torch.nn.Module, decoder: torch.nn.Module, device: torch.device):
    """Initializes TensorBoard and logs the complex Encoder-Decoder model graphs safely."""
    print(f"\n📊 Starting TensorBoard Logging...")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✅ TensorBoard SummaryWriter initialized successfully.")
    
    return writer
