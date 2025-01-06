import numpy as np
import torch
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalNormalizedDCG,
    RetrievalMAP,
    RetrievalMRR,
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryAUROC
)
from typing import List


def ndcg_at_k(r: np.ndarray, k: int):
    r = np.asfarray(r)[:k]
    if r.size:
        dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        idcg = np.sum(np.ones_like(r) / np.log2(np.arange(2, r.size + 2)))
        return dcg / idcg
    return 0.


class ScoreMetricHandler():
    def __init__(
        self,
        device: torch.device
    ):
        self.pos_scores: List[torch.Tensor] = []
        self.neg_scores: List[torch.Tensor] = []
        self.is_calculated = False
        self.result = None
        self.device = device

    def update(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        self.pos_scores.append(pos_scores.clone().detach()).to(self.device)
        self.neg_scores.append(neg_scores.clone().detach()).to(self.device)
        self.is_calculated = False

    def compute(self):
        if not self.is_calculated:
            pos_scores = torch.cat(self.pos_scores)
            neg_scores = torch.cat(self.neg_scores)
            self.pos_mean = pos_scores.mean().item()
            self.pos_min = pos_scores.min().item()
            self.pos_max = pos_scores.max().item()
            self.pos_std = pos_scores.std().item()
            self.neg_mean = neg_scores.mean().item()
            self.neg_min = neg_scores.min().item()
            self.neg_max = neg_scores.max().item()
            self.neg_std = neg_scores.std().item()
            self.diff_mean = self.pos_mean - self.neg_mean

            self.is_calculated = True

        return {
            "pos_mean": self.pos_mean,
            "pos_min": self.pos_min,
            "pos_max": self.pos_max,
            "pos_std": self.pos_std,
            "neg_mean": self.neg_mean,
            "neg_min": self.neg_min,
            "neg_max": self.neg_max,
            "neg_std": self.neg_std,
            "diff_mean": self.diff_mean,
        }

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v, num_round)
            for k, v in self.compute().items()
        }


class MetricsHandler():
    """
    入力を繰り返し受け取り、最終的な計算を行う。
    """
    def __init__(self, device, threshold: float = 0.5):
        self.threshold = threshold
        self.device = device
        self.reset()

    def reset(self):
        self.probas = []
        self.targets = []
        self.user_indices = []
        self.is_calculated = False

    def update(self,
               probas: torch.Tensor,
               targets: torch.Tensor,
               user_indices: torch.Tensor):
        probas = probas.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)
        user_indices = user_indices.to(self.device)
        self.probas.append(probas)
        self.targets.append(targets)
        self.user_indices.append(user_indices)

    def compute(self):
        if not self.is_calculated:
            all_probas = torch.cat(self.probas)
            all_targets = torch.cat(self.targets)
            all_user_indices = torch.cat(self.user_indices)

            recall_at_10 = RetrievalRecall(empty_target_action="skip", top_k=10).to(self.device)
            recall_at_20 = RetrievalRecall(empty_target_action="skip", top_k=20).to(self.device)
            pre_at_10 = RetrievalPrecision(empty_target_action="skip", top_k=10, adaptive_k=True).to(self.device)  # noqa: E501
            pre_at_20 = RetrievalPrecision(empty_target_action="skip", top_k=20, adaptive_k=True).to(self.device)  # noqa: E501
            ndcg_at_10 = RetrievalNormalizedDCG(empty_target_action="skip", top_k=10).to(self.device)  # noqa: E501
            ndcg_at_20 = RetrievalNormalizedDCG(empty_target_action="skip", top_k=20).to(self.device)  # noqa: E501
            map_at_10 = RetrievalMAP(empty_target_action="skip", top_k=10).to(self.device)
            map_at_20 = RetrievalMAP(empty_target_action="skip", top_k=20).to(self.device)
            mrr_at_10 = RetrievalMRR(empty_target_action="skip", top_k=10).to(self.device)
            mrr_at_20 = RetrievalMRR(empty_target_action="skip", top_k=20).to(self.device)
            binary_accuracy = BinaryAccuracy(threshold=self.threshold).to(self.device)
            binary_recall = BinaryRecall(threshold=self.threshold).to(self.device)
            binary_f1 = BinaryF1Score(threshold=self.threshold).to(self.device)
            binary_confusion_matrix = BinaryConfusionMatrix(threshold=self.threshold)
            binary_confusion_matrix.to(self.device)
            binary_aucroc = BinaryAUROC().to(self.device)

            self.recall_at_10_result = recall_at_10(all_probas, all_targets, all_user_indices)
            self.recall_at_20_result = recall_at_20(all_probas, all_targets, all_user_indices)
            self.pre_at_10_result = pre_at_10(all_probas, all_targets, all_user_indices)
            self.pre_at_20_result = pre_at_20(all_probas, all_targets, all_user_indices)
            self.ndcg_at_10_result = ndcg_at_10(all_probas, all_targets, all_user_indices)
            self.ndcg_at_20_result = ndcg_at_20(all_probas, all_targets, all_user_indices)
            self.map_at_10_result = map_at_10(all_probas, all_targets, all_user_indices)
            self.map_at_20_result = map_at_20(all_probas, all_targets, all_user_indices)
            self.mrr_at_10_result = mrr_at_10(all_probas, all_targets, all_user_indices)
            self.mrr_at_20_result = mrr_at_20(all_probas, all_targets, all_user_indices)
            self.binary_accuracy_result = binary_accuracy(all_probas, all_targets)
            self.binary_recall_result = binary_recall(all_probas, all_targets)
            self.binary_f1_result = binary_f1(all_probas, all_targets)
            self.binary_confusion_matrix_result = binary_confusion_matrix(all_probas, all_targets)
            self.binary_aucroc_result = binary_aucroc(all_probas, all_targets)

            self.is_calculated = True

        return {
            "recall@10": self.recall_at_10_result.item(),
            "recall@20": self.recall_at_20_result.item(),
            "precision@10": self.pre_at_10_result.item(),
            "precision@20": self.pre_at_20_result.item(),
            "ndcg@10": self.ndcg_at_10_result.item(),
            "ndcg@20": self.ndcg_at_20_result.item(),
            "map@10": self.map_at_10_result.item(),
            "map@20": self.map_at_20_result.item(),
            "mrr@10": self.mrr_at_10_result.item(),
            "mrr@20": self.mrr_at_20_result.item(),
            "accuracy": self.binary_accuracy_result.item(),
            "recall": self.binary_recall_result.item(),
            "AUROC": self.binary_aucroc_result.item(),
            "f1": self.binary_f1_result.item(),
            "tn": self.binary_confusion_matrix_result[0][0].item(),
            "fp": self.binary_confusion_matrix_result[0][1].item(),
            "fn": self.binary_confusion_matrix_result[1][0].item(),
            "tp": self.binary_confusion_matrix_result[1][1].item(),
        }

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v, num_round)
            for k, v in self.compute().items()
        }
