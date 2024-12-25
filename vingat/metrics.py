import numpy as np
import torch
from typing import List
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision, RetrievalAUROC, RetrievalNormalizedDCG
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, ConfusionMatrix


def ndcg_at_k(r: np.ndarray, k: int):
    r = np.asfarray(r)[:k]
    if r.size:
        dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        idcg = np.sum(np.ones_like(r) / np.log2(np.arange(2, r.size + 2)))
        return dcg / idcg
    return 0.


def score_stastics(pos_scores: List[torch.Tensor], neg_scores: List[torch.Tensor]):

    # torch.catで結合 (1次元テンソルを想定)
    u_pos_scores_tensor = torch.cat(pos_scores)
    u_neg_scores_tensor = torch.cat(neg_scores)

    # NumPy配列に変換
    u_pos_scores_np = u_pos_scores_tensor.detach().cpu().numpy()
    u_neg_scores_np = u_neg_scores_tensor.detach().cpu().numpy()

    # 統計量を算出
    pos_mean = np.mean(u_pos_scores_np)
    pos_min = np.min(u_pos_scores_np)
    pos_max = np.max(u_pos_scores_np)
    pos_std = np.std(u_pos_scores_np)

    neg_mean = np.mean(u_neg_scores_np)
    neg_min = np.min(u_neg_scores_np)
    neg_max = np.max(u_neg_scores_np)
    neg_std = np.std(u_neg_scores_np)

    diff_mean = pos_mean - neg_mean

    return {
        "data": {
            "score_metrics/pos_mean": pos_mean,
            "score_metrics/pos_min": pos_min,
            "score_metrics/pos_max": pos_max,
            "score_metrics/pos_std": pos_std,
            "score_metrics/neg_mean": neg_mean,
            "score_metrics/neg_min": neg_min,
            "score_metrics/neg_max": neg_max,
            "score_metrics/neg_std": neg_std,
            "score_metrics/diff_mean": diff_mean,
        },
    }


class MetricsAtK():
    def __init__(self, k: int):
        self.k = k
        self.recall = RetrievalRecall(top_k=k)
        self.precision = RetrievalPrecision(top_k=k, adaptive_k=True)
        self.auroc = RetrievalAUROC(top_k=k)
        self.ndcg = RetrievalNormalizedDCG(top_k=k)

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexed: torch.Tensor):
        self.recall.update(preds, target, indexed)
        self.precision.update(preds, target, indexed)
        self.auroc.update(preds, target, indexed)
        self.ndcg.update(preds, target, indexed)
        return self

    def compute(self, prefix):
        return {
            f"{prefix}recall": self.recall.compute(),
            f"{prefix}precision": self.precision.compute(),
            f"{prefix}auroc": self.auroc.compute(),
            f"{prefix}ndcg": self.ndcg.compute()
        }


class Metrics():
    def __init__(self, k: int):
        self.k = k
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)
        self.f1 = BinaryF1Score(threshold=0.5)
        self.confusion_matrix = ConfusionMatrix(num_classes=2)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.accuracy.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)
        self.confusion_matrix.update(preds, target)
        return self

    def compute(self, prefix: str):
        cm = self.confusion_matrix.compute()
        return {
            f"{prefix}accuracy": self.accuracy.compute(),
            f"{prefix}precision": self.precision.compute(),
            f"{prefix}recall": self.recall.compute(),
            f"{prefix}f1": self.f1.compute(),
            f"{prefix}true_negative": cm[0][0],
            f"{prefix}false_positive": cm[0][1],
            f"{prefix}false_negative": cm[1][0],
            f"{prefix}true_positive": cm[1][1]
        }
