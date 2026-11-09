import re
import torch
from typing import List, Dict, Union

class AtomLevelSmilesTokenizer:
    def __init__(self):
        # The industry-standard regex for splitting SMILES into chemical units
        # Position: token.py (Line 8)
        self.regex_pattern = r"(\[[^\]]+\]|Br?|Cl?|H|B|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.regex_pattern)
        
        # Define foundational special tokens
        self.pad_token = "[PAD]"  # Padding token to fill shapes
        self.sos_token = "[SOS]"  # Start of Sequence marker
        self.eos_token = "[EOS]"  # End of Sequence marker
        self.unk_token = "[UNK]"  # Unknown token fallback
        
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # Maps initialized during compilation
        self.stoi: Dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.itos: Dict[int, str] = {idx: token for idx, token in enumerate(self.special_tokens)}
        self.vocab_size = len(self.special_tokens)

    def tokenize_string(self, smiles_str: str) -> List[str]:
        """Splits a SMILES string into chemical sub-tokens using the regex pattern."""
        return [token for token in self.regex.findall(smiles_str)]

    def fit_on_dataframe(self, smiles_column) -> None:
        """Scans the entire dataset column to discover every unique chemical element."""
        unique_tokens = set()
        for smiles in smiles_column:
            if isinstance(smiles, str):
                unique_tokens.update(self.tokenize_string(smiles))
        
        # Build the structured vocabulary lists
        all_vocab = self.special_tokens + sorted(list(unique_tokens))
        
        self.stoi = {token: idx for idx, token in enumerate(all_vocab)}
        self.itos = {idx: token for idx, token in enumerate(all_vocab)}
        self.vocab_size = len(all_vocab)
        
        print(f"SMILES Tokenizer compiled successfully. Unique Vocabulary Size: {self.vocab_size}")

    def encode(self, smiles_str: str, max_length: int = 121, **kwargs) -> torch.Tensor:
        """Converts text strings into integer tensors with structural markers and padding."""
        raw_tokens = self.tokenize_string(smiles_str)
        
        # Structure the sequence with system tokens
        # [SOS] + [Tokens...] + [EOS]
        tokens = [self.sos_token] + raw_tokens + [self.eos_token]
        
        # Map tokens to their respective numerical index identifiers
        token_ids = []
        for token in tokens:
            token_ids.append(self.stoi.get(token, self.stoi[self.unk_token]))
            
        # Truncate if sequence exceeds your maximum threshold limit
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.stoi[self.eos_token]]
            
        # Dynamically append padding vectors up to the max_length constraint boundary
        padding_needed = max_length - len(token_ids)
        if padding_needed > 0:
            token_ids += [self.stoi[self.pad_token]] * padding_needed
            
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Translates integer tensors back into human-readable chemical strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        decoded_tokens = []
        for idx in token_ids:
            token = self.itos.get(idx, self.unk_token)
            # Stop parsing immediately if we reach padding or end of sequence indicators
            if token == self.eos_token or token == self.pad_token:
                break
            if token != self.sos_token:
                decoded_tokens.append(token)
                
        return "".join(decoded_tokens)
