import numpy as np
import torch
from torchmetrics import MetricCollection
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
from typing import Dict, List


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
        self.pos_scores.append(pos_scores.clone().detach().to(self.device))
        self.neg_scores.append(neg_scores.clone().detach().to(self.device))
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

            collection = MetricCollection({
                "recall@10": RetrievalRecall(empty_target_action="skip", top_k=10),
                "recall@20": RetrievalRecall(empty_target_action="skip", top_k=20),
                "precision@10": RetrievalPrecision(empty_target_action="skip", top_k=10, adaptive_k=True),  # noqa: E501
                "precision@20": RetrievalPrecision(empty_target_action="skip", top_k=20, adaptive_k=True),  # noqa: E501
                "ndcg@10": RetrievalNormalizedDCG(empty_target_action="skip", top_k=10),  # noqa: E501
                "ndcg@20": RetrievalNormalizedDCG(empty_target_action="skip", top_k=20),  # noqa: E501
                "map@10": RetrievalMAP(empty_target_action="skip", top_k=10),
                "map@20": RetrievalMAP(empty_target_action="skip", top_k=20),
                "mrr@10": RetrievalMRR(empty_target_action="skip", top_k=10),
                "mrr@20": RetrievalMRR(empty_target_action="skip", top_k=20),
                "accuracy": BinaryAccuracy(threshold=self.threshold),
                "recall": BinaryRecall(threshold=self.threshold),
                "f1": BinaryF1Score(threshold=self.threshold),
                "cm": BinaryConfusionMatrix(threshold=self.threshold),
                "AUROC": BinaryAUROC(),
            }).to(self.device)

            result = collection(all_probas, all_targets, indexes=all_user_indices)
            result["tn"] = result["cm"][0][0].type(torch.float16)
            result["fp"] = result["cm"][0][1].type(torch.float16)
            result["fn"] = result["cm"][1][0].type(torch.float16)
            result["tp"] = result["cm"][1][1].type(torch.float16)
            del result["cm"]

            self.result = result

            self.is_calculated = True
        return self.result

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v.item(), num_round)
            for k, v in self.compute().items()
        }


class MetricsHandlerForUserLoop():
    def __init__(self, device, threshold: float = 0.5):
        self.threshold = threshold
        self.device = device
        self.results: Dict[str, List[torch.Tensor]] = {}

    def update(self,
               probas: torch.Tensor,
               targets: torch.Tensor,
               user_indices: torch.Tensor):
        mh = MetricsHandler(self.device, threshold=self.threshold)
        mh.update(probas, targets, user_indices)
        result = mh.compute()
        for k, v in result.items():
            if k not in self.results.keys():
                self.results[k] = []
            self.results[k].append(v.unsqueeze(0))

    def compute(self):
        print(self.results.items())
        return {
            k: torch.cat(v).mean().item()
            for k, v in self.results.items()
        }

    def log(self, prefix: str = "", separator: str = "/", num_round: int = 8):
        return {
            f"{prefix}{separator}{k}": round(v, num_round)
            for k, v in self.compute().items()
        }
