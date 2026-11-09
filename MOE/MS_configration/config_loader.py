from pathlib import Path
import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = r"/PROJECT_PIMPI/MOE/MS_configration/config.yaml"):
        # Resolve project root (2 levels up from this file, or adjust as needed)
        self.base_dir = Path(__file__).resolve().parents[2]
        
        # Load YAML file
        full_path = self.base_dir / config_path
        if not full_path.exists():
            # Fallback to current directory if not found in parents
            full_path = Path(config_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Master configuration file not found at: {full_path}")
            
        with open(full_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    # ==========================================
    # --- General Settings ---
    # ==========================================
    @property
    def device(self) -> str:
        return self.cfg["general"]["device"]

    @property
    def seed(self) -> int:
        return self.cfg["general"]["seed"]

    # ==========================================
    # --- File & Directory Paths ---
    # ==========================================
    @property
    def data_dir(self) -> str:
        return self.cfg["paths"]["data_dir"]

    @property
    def full_data_path(self) -> Path:
        p = self.cfg["paths"]
        return self.base_dir / p["data_dir"] / p["full_data_filename"]

    @property
    def subset_data_path(self) -> Path:
        p = self.cfg["paths"]
        return self.base_dir / p["data_dir"] / p["subset_filename"]

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / self.cfg["paths"]["checkpoint_dir"]

    @property
    def tokenizer_path(self) -> Path:
        return self.checkpoint_dir / self.cfg["paths"]["tokenizer_filename"]

    @property
    def best_params_path(self) -> Path:
        return self.checkpoint_dir / self.cfg["paths"]["best_params_filename"]

    # ==========================================
    # --- Tokenizer Settings ---
    # ==========================================
    @property
    def max_smiles_len(self) -> int:
        return self.cfg["tokenizer"]["max_smiles_len"]

    # ==========================================
    # --- Model: Adaptive Spectral Encoder ---
    # ==========================================
    @property
    def enc_peak_hidden_dim(self) -> int:
        return self.cfg["adaptive_encoder"]["peak_hidden_dim"]

    @property
    def enc_max_peaks(self) -> int:
        return self.cfg["adaptive_encoder"]["max_peaks"]

    @property
    def enc_embed_dim(self) -> int:
        return self.cfg["adaptive_encoder"]["embed_dim"]

    @property
    def enc_latent_dim(self) -> int:
        return self.cfg["adaptive_encoder"]["latent_dim"]

    @property
    def enc_num_heads(self) -> int:
        return self.cfg["adaptive_encoder"]["num_heads"]

    @property
    def enc_dropout(self) -> float:
        return self.cfg["adaptive_encoder"]["dropout"]

    @property
    def enc_act_type(self) -> str:
        return self.cfg["adaptive_encoder"]["act_type"]

    # ==========================================
    # --- Model: Structural SMILES Decoder ---
    # ==========================================
    @property
    def dec_embed_dim(self) -> int:
        return self.cfg["structural_decoder"]["embed_dim"]

    @property
    def dec_num_layers(self) -> int:
        return self.cfg["structural_decoder"]["num_layers"]

    @property
    def dec_num_heads(self) -> int:
        return self.cfg["structural_decoder"]["num_heads"]

    @property
    def dec_dropout(self) -> float:
        return self.cfg["structural_decoder"]["dropout"]

    @property
    def dec_act_type(self) -> str:
        return self.cfg["structural_decoder"]["act_type"]

    @property
    def max_seq_len(self) -> int:
        return self.cfg["structural_decoder"]["max_seq_len"]

    @property
    def num_memory_tokens(self) -> int:
        return self.cfg["structural_decoder"]["num_memory_tokens"]

    # ==========================================
    # --- Training Settings ---
    # ==========================================
    @property
    def train_batch_size(self) -> int:
        return self.cfg["training"]["batch_size"]

    @property
    def val_batch_size(self) -> int:
        return self.cfg["training"]["val_batch_size"]

    @property
    def train_epochs(self) -> int:
        return self.cfg["training"]["epochs"]

    @property
    def train_learning_rate(self) -> float:
        return self.cfg["training"]["learning_rate"]

    @property
    def train_weight_decay(self) -> float:
        return self.cfg["training"]["weight_decay"]

    @property
    def max_grad_norm(self) -> float:
        return self.cfg["training"]["max_grad_norm"]

    # ==========================================
    # --- Hyperparameter Optimization (HPO) ---
    # ==========================================
    @property
    def hpo_subset_size(self) -> int:
        return self.cfg["hpo"]["subset_size"]

    @property
    def hpo_epochs(self) -> int:
        return self.cfg["hpo"]["epochs"]

    @property
    def hpo_n_trials(self) -> int:
        return self.cfg["hpo"]["n_trials"]

    def get_search_space(self, trial):
        """Generates Optuna trial suggestions dynamically from the YAML search space."""
        space = self.cfg["hpo"]["search_space"]
        return {
            "learning_rate": trial.suggest_float("learning_rate", 
                           space["learning_rate"]["min"], 
                           space["learning_rate"]["max"], 
                           log=space["learning_rate"]["log"]),

            "batch_size": trial.suggest_categorical("batch_size", 
                           space["batch_size"]),

            "weight_decay": trial.suggest_float("weight_decay", 
                            space["weight_decay"]["min"], 
                            space["weight_decay"]["max"], 
                            log=space["weight_decay"]["log"]),

            "embed_dim": trial.suggest_categorical("embed_dim", 
                         space["embed_dim"]),

            "latent_dim": trial.suggest_categorical("latent_dim", 
                          space["latent_dim"]),

            "num_layers": trial.suggest_int("num_layers", 
                          space["num_layers"][0], 
                          space["num_layers"][1]),

            "num_heads": trial.suggest_categorical("num_heads", 
                         space["num_heads"]),

            "max_peaks": trial.suggest_categorical("max_peaks", 
                         space["max_peaks"]),

            "dropout": trial.suggest_float("dropout", 
                        space["dropout"]["min"], 
                        space["dropout"]["max"]),
            
        }

    # ==========================================
    # --- Raw Dict Accessors (for flexibility) ---
    # ==========================================
    @property
    def encoder_config(self) -> Dict[str, Any]:
        return self.cfg["adaptive_encoder"]

    @property
    def decoder_config(self) -> Dict[str, Any]:
        return self.cfg["structural_decoder"]

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.cfg["training"]

    @property
    def hpo_settings(self) -> Dict[str, Any]:
        return self.cfg["hpo"]


# Global singleton config instance for easy importing across your project
config = ConfigManager()