import numpy as np
import torch
from typing import List
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
    BinaryConfusionMatrix
)


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


class MetricsHandler():
    """
    入力を繰り返し受け取り、最終的な計算を行う。
    """
    def __init__(self, threshold: float = 0.5):
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
        self.probas.append(probas)
        self.targets.append(targets)
        self.user_indices.append(user_indices)

    def compute(self):
        if not self.is_calculated:
            all_probas = torch.cat(self.probas)
            all_targets = torch.cat(self.targets)
            all_user_indices = torch.cat(self.user_indices)

            recall_at_10 = RetrievalRecall(empty_target_action="skip", top_k=10)
            recall_at_20 = RetrievalRecall(empty_target_action="skip", top_k=20)
            pre_at_10 = RetrievalPrecision(empty_target_action="skip", top_k=10, adaptive_k=True)
            pre_at_20 = RetrievalPrecision(empty_target_action="skip", top_k=20, adaptive_k=True)
            ndcg_at_10 = RetrievalNormalizedDCG(empty_target_action="skip", top_k=10)
            ndcg_at_20 = RetrievalNormalizedDCG(empty_target_action="skip", top_k=20)
            map_at_10 = RetrievalMAP(empty_target_action="skip", top_k=10)
            map_at_20 = RetrievalMAP(empty_target_action="skip", top_k=20)
            mrr_at_10 = RetrievalMRR(empty_target_action="skip", top_k=10)
            mrr_at_20 = RetrievalMRR(empty_target_action="skip", top_k=20)
            binary_accuracy = BinaryAccuracy(threshold=self.threshold)
            binary_recall = BinaryRecall(threshold=self.threshold)
            binary_f1 = BinaryF1Score(threshold=self.threshold)
            binary_confusion_matrix = BinaryConfusionMatrix(threshold=self.threshold)

            self.recall_at_10_result = recall_at_10(all_probas, all_targets, all_user_indices)
            self.recall_at_20_result = recall_at_20(all_probas, all_targets, all_user_indices)
            self.pre_at_10_result = pre_at_10(all_probas, all_targets, all_user_indices)
            self.pre_at_20_result = pre_at_20(all_probas, all_targets, all_user_indices)
            self.ndcg_at_10_result = ndcg_at_10(all_probas, all_targets, all_user_indices)
            self.ndcg_at_20_result = ndcg_at_20(all_probas, all_targets, all_user_indices)
            self.mat_at_10_result = map_at_10(all_probas, all_targets, all_user_indices)
            self.mat_at_20_result = map_at_20(all_probas, all_targets, all_user_indices)
            self.mrr_at_10_result = mrr_at_10(all_probas, all_targets, all_user_indices)
            self.mrr_at_20_result = mrr_at_20(all_probas, all_targets, all_user_indices)
            self.binary_accuracy_result = binary_accuracy(all_probas, all_targets)
            self.binary_recall_result = binary_recall(all_probas, all_targets)
            self.binary_f1_result = binary_f1(all_probas, all_targets)
            self.binary_confusion_matrix_result = binary_confusion_matrix(all_probas, all_targets)

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
