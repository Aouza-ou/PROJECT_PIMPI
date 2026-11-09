from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "MOE" / "DATA"
FULL_DATA_PATH = DATA_DIR / "train.parquet"
SUBSET_DATA_PATH = DATA_DIR / "train_part_1.parquet"  # Your 500-row subset
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.pt"

# --- Fixed HPO Settings ---
HPO_SUBSET_SIZE = 50  # Your 500-row subset
HPO_EPOCHS = 3
MAX_SMILES_LEN = 121   # Your original config
DEVICE = "cuda"


# --- Optuna Search Space ---
def get_search_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),  # Reduced from [16, 32, 64]
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "embed_dim": trial.suggest_categorical("embed_dim", [128, 256]),  # Reduced from [256, 512]
        "latent_dim": trial.suggest_categorical("latent_dim", [128, 256]),  # Reduced from [256, 512]
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "max_peaks": trial.suggest_categorical("max_peaks", [256, 512]),  # Reduced from [512, 1024, 2000]
    }
""" 
# --- Optuna Search Space ---
def get_search_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "embed_dim": trial.suggest_categorical("embed_dim", [256, 512]),
        "latent_dim": trial.suggest_categorical("latent_dim", [256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "max_peaks": trial.suggest_categorical("max_peaks", [512, 1024, 2000]), 
    }
"""

