import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import pandas as pd


class StaticEmbeddingLoader():
    def __init__(self, data: pd.DataFrame, dimention: int, device):
        self.data = data
        self.embedding_dim = dimention
        self.cols = [f"e_{i}" for i in range(dimention)]
        self.device = device

    def __call__(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, self.cols].values
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device
        )


class TextEncoder(nn.Module):
    def __init__(self, dimention: int, device) -> None:
        super().__init__()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                                         truncate_dim=dimention,
                                         device=device)
        self.device = device

    def forward(self, sentences: list):
        return torch.as_tensor(
            self.sbert.encode(sentences),
            dtype=torch.float32,
            device=self.device
        )


class CookingDirectionEncoder(nn.Module):
    def __init__(self, data: pd.DataFrame, dimention: int, device) -> None:
        super().__init__()
        self.device = device
        self.data = data
        self.sbert = TextEncoder(dimention, device)

    def forward(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, "direction"].values
        return torch.as_tensor(
            self.sbert(values),
            dtype=torch.float32,
            device=self.device
        )


class VLMEncoder(nn.Module):
    def __init__(self, data: pd.DataFrame, dimention: int, device) -> None:
        super().__init__()
        self.data = data
        self.embedding_dim = dimention
        self.device = device
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                                         truncate_dim=dimention,
                                         device=device)

    def forward(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices].values.reshape(-1)
        return torch.as_tensor(
            self.sbert.encode(values),
            dtype=torch.float32,
            device=self.device
        )