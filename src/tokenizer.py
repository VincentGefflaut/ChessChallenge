from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

class ChessTokenizer(PreTrainedTokenizer):
    """
    Tokenizer déterministe au niveau 'case' (Square-level).
    Compatible avec les scripts de train/data du projet Chess Challenge.
    """
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Tokens spéciaux identiques au projet original
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab_file: Optional[str] = None, vocab: Optional[Dict[str, int]] = None, **kwargs):
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        # Nettoyage des kwargs pour éviter les doublons lors de l'init parent
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)

        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._create_square_vocab()
            
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs
        )

    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "",
        split: str = "",
        column: str = "",
        min_frequency: int = 0,
        max_samples: Optional[int] = None,
    ) -> "ChessTokenizer":
        """
        Méthode de compatibilité. 
        Pour le SquareTokenizer, le vocabulaire est fixe, 
        on ignore donc les arguments et on retourne une instance standard.
        """
        print("Square Tokenizer: Using fixed deterministic vocabulary.")
        return cls()

    def _create_square_vocab(self) -> Dict[str, int]:
        """Crée le vocabulaire fixe de cases (64) + promos (4) + spéciaux (4)."""
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        squares = [f + r for f in files for r in ranks]
        promotions = ['q', 'r', 'b', 'n']
        
        all_tokens = special_tokens + squares + promotions
        return {token: idx for idx, token in enumerate(all_tokens)}

    # --- MÉTHODES REQUISES POUR HUGGING FACE COMPATIBILITY ---
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        """Découpe 'WPe2e4' en ['e2', 'e4']."""
        moves = text.strip().split()
        tokens = []
        for m in moves:
            if m in {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}:
                tokens.append(m)
                continue
            
            # Nettoyage Regex : on ne garde que les coordonnées a-h, 1-8 et promos qrbn
            clean_m = re.sub(r'[\(\)x\+\*WBPNBRQK]', '', m)
            
            if len(clean_m) >= 4:
                tokens.append(clean_m[0:2]) # Case départ
                tokens.append(clean_m[2:4]) # Case arrivée
                if len(clean_m) == 5:
                    tokens.append(clean_m[4]) # Promotion
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Utile pour reconstruire le format texte si besoin
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (f"{filename_prefix}-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)