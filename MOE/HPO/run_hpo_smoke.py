import sys
import optuna
import json
import torch
from pathlib import Path
from tqdm import tqdm

SMOKE_TEST=True
SMOKE_TRIALS = 4
SMOKE_SAMPLES = 500
SMOKE_EPOCHS = 4 



from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 1. IMPORTS: Use your new config system
from MOE.MS_configration.config_loader import config
from MOE.HPO.hpo_data import prepare_tokenizer_and_subset, get_hpo_dataloaders
from MOE.HPO.hpo_model import build_model_and_optimizer
from MOE.HPO.hpo_engine import run_hpo_epoch
import gc

def main():
    # 1. Prepare Data & Tokenizer ONCE before Optuna starts
    print("Initializing Tokenizer and HPO Subset...")
    tokenizer = prepare_tokenizer_and_subset()
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.stoi[tokenizer.pad_token]
    
    # Pull HPO settings from config.yaml
    hpo_cfg = config.hpo_settings
    max_grad_norm = config.training_config.get('max_grad_norm', 1.0)
    device = torch.device(config.device)

# Shrinking the Data for prevent to explode 
    if SMOKE_TEST:
        hpo_cfg['n_trials'] = SMOKE_TRIALS
        hpo_cfg['epochs'] = SMOKE_EPOCHS
        hpo_cfg['subset_size'] = SMOKE_SAMPLES
        print(f"*** SMOKE TEST MODE {SMOKE_TRIALS} trials, {SMOKE_SAMPLES} sample {SMOKE_EPOCHS} epoch ")


    def objective(trial):
        """Optuna Objective Function"""
        # 2. Get suggested hyperparameters using the config-driven method
        hp = config.get_search_space(trial)

        # Clear GPU memory before building new model
        torch.cuda.empty_cache()
        gc.collect()

        
        # 3. Build DataLoaders 
        subset_size = hpo_cfg.get('subset_size', None) if SMOKE_TEST else None # Check subset is true (1) or not (0)
        train_loader, val_loader = get_hpo_dataloaders(
            tokenizer, 
            batch_size=hp["batch_size"], 
            max_peaks=hp["max_peaks"],
            subset_size=subset_size
        )
        
        # 4. Build Model
        encoder, decoder, optimizer, criterion, all_parameters = build_model_and_optimizer(
            hp, vocab_size, pad_idx, device
        )
        
        # 5. Train for HPO_EPOCHS (from config.yaml)
        best_val_loss = float("inf")
        epoch_pbar = tqdm(range(hpo_cfg['epochs']), desc=f"Trial {trial.number + 1} Training", leave=False)
        
        for epoch in epoch_pbar:
            val_loss = run_hpo_epoch(
                encoder, decoder, train_loader, val_loader, 
                criterion, optimizer, device, pad_idx, vocab_size, all_parameters, max_norm=max_grad_norm
            )
            
            epoch_pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})
            # Report intermediate value to Optuna for pruning (kills bad trials early!)
            trial.report(val_loss, epoch)
            if trial.should_prune():

                # ==== VRAM protection ====
                # MUST CLEAN UP BEFORE PRUNING!
                del encoder, decoder, criterion, all_parameters
                del train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.exceptions.TrialPruned()
                
            best_val_loss = min(best_val_loss, val_loss)
        # CRITICAL: Delete models BEFORE next trial starts
        del encoder, decoder, criterion, all_parameters
        del train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

            
        return best_val_loss

    # 6. Setup Optuna Study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    )
    
    print(f"Starting Optuna HPO ({hpo_cfg['n_trials']} trials)...")
    study.optimize(objective, n_trials=hpo_cfg['n_trials'], show_progress_bar=True)
    
    # 7. Save the winning hyperparameters
    best_params = study.best_params
    best_params["max_smiles_len"] = config.max_smiles_len #['max_smiles_len'] 
    
    config.checkpoint_dir.mkdir(exist_ok=True)
    best_params_path = config.checkpoint_dir / config.cfg["paths"]["best_params_filename"]
    
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"✅ HPO Complete! Best Validation Loss: {study.best_value:.4f}")
    print(f"✅ Best Hyperparameters saved to: {best_params_path}")
    print(best_params)

if __name__ == "__main__":
    main()