import sys
from pathlib import Path

# 1. Automatically find the root 'MOE' folder path
MOE_ROOT = Path(__file__).resolve().parent

# 2. Register all sub-folder locations (The Registry)
ENCODER_DIR = MOE_ROOT / "Encoder" / "embadding"
DECODER_DIR = MOE_ROOT / "Decoder" / "smiles_decoder"
TOKENIZER_DIR = MOE_ROOT / "Tokennizer" / "smiles_tokenizer"

# 3. Automatically tell Python to look inside these folders
for folder in [MOE_ROOT, ENCODER_DIR, DECODER_DIR, TOKENIZER_DIR]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
