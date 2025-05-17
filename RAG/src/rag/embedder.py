from __future__ import annotations
from typing import Sequence, List
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer
from .config import EmbedderConfig

class SentenceTransformerEmbedder:
    def __init__(self, cfg:EmbedderConfig):
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.batch_size = cfg.batch_size

    def encode(self, texts:Sequence[str]) -> torch.Tensor:
        return self.model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False)
