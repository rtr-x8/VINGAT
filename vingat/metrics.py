import numpy as np
import torch
from typing import List
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalAUROC,
    RetrievalNormalizedDCG
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    ConfusionMatrix
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


class MetricsAtK():
    def __init__(self, k: int, device: torch.device):
        self.k = k
        self.device = device
        self.recall = RetrievalRecall(top_k=k).to(device)
        self.precision = RetrievalPrecision(top_k=k, adaptive_k=True).to(device)
        self.auroc = RetrievalAUROC(top_k=k).to(device)
        self.ndcg = RetrievalNormalizedDCG(top_k=k).to(device)

        # One
        self.accuracy = BinaryAccuracy(threshold=0.0).to(device)
        self.f1 = BinaryF1Score(threshold=0.0).to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexed: torch.Tensor):
        preds = preds.to(self.device)
        target = target.to(self.device)
        indexed = indexed.to(self.device)
        self.recall.update(preds, target, indexed)
        self.precision.update(preds, target, indexed)
        self.auroc.update(preds, target, indexed)
        self.ndcg.update(preds, target, indexed)
        self.accuracy.update(preds, target)
        self.f1.update(preds, target)
        return self

    def compute(self, prefix, suffix):
        return {
            f"{prefix}recall{suffix}": self.recall.compute().item(),
            f"{prefix}precision{suffix}": self.precision.compute().item(),
            f"{prefix}auroc{suffix}": self.auroc.compute().item(),
            f"{prefix}ndcg{suffix}": self.ndcg.compute().item(),
            f"{prefix}accuracy{suffix}": self.accuracy.compute().item(),
            f"{prefix}f1{suffix}": self.f1.compute().item()
        }


class MetricsAll():
    def __init__(self, device: torch.device):
        self.device = device
        self.accuracy = BinaryAccuracy(threshold=0.0).to(device)
        self.precision = BinaryPrecision(threshold=0.0).to(device)
        self.recall = BinaryRecall(threshold=0.0).to(device)
        self.f1 = BinaryF1Score(threshold=0.0).to(device)
        self.confusion_matrix = ConfusionMatrix(task="binary", num_classes=2).to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.to(self.device)
        target = target.to(self.device)
        self.accuracy.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)
        self.confusion_matrix.update(preds, target)
        return self

    def compute(self, prefix: str):
        cm = self.confusion_matrix.compute().cpu().numpy()
        return {
            f"{prefix}accuracy": self.accuracy.compute().item(),
            f"{prefix}precision": self.precision.compute().item(),
            f"{prefix}recall": self.recall.compute().item(),
            f"{prefix}f1": self.f1.compute().item(),
            f"{prefix}true_negative": cm[0][0],
            f"{prefix}false_positive": cm[0][1],
            f"{prefix}false_negative": cm[1][0],
            f"{prefix}true_positive": cm[1][1]
        }


class BatchMetricsAverager:
    """
    バッチごとにMetricsAllを使って混同行列やaccuracy等を計算し、
    エポックが終わるタイミングでバッチ平均のメトリクスを取得するクラス。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        """
        バッチ集計を初期化する
        """
        self.batch_accuracy_sum = 0.0
        self.batch_precision_sum = 0.0
        self.batch_recall_sum = 0.0
        self.batch_f1_sum = 0.0
        self.n_batches = 0
        self.pos_count = 0
        self.neg_count = 0

    def update(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        バッチ単位で MetricsAll を使い、その結果を足し合わせる。
        pos_scores, neg_scores はモデルの出力（スコア）。
        """
        # このバッチだけのメトリクスオブジェクトを生成
        metrics_batch = MetricsAll(self.device)

        # Positive + Negative をまとめて update
        metrics_batch.update(
            preds=torch.cat([pos_scores, neg_scores], dim=0),
            target=torch.tensor(
                [1] * len(pos_scores) + [0] * len(neg_scores),
                dtype=torch.long,
                device=self.device
            )
        )
        # バッチごとの結果を計算
        batch_result = metrics_batch.compute(prefix="")

        # 次のバッチ時に干渉しないようリセット
        metrics_batch.confusion_matrix.reset()

        # 結果を集計
        self.batch_accuracy_sum += batch_result["accuracy"]
        self.batch_precision_sum += batch_result["precision"]
        self.batch_recall_sum += batch_result["recall"]
        self.batch_f1_sum += batch_result["f1"]
        self.n_batches += 1
        self.pos_count += len(pos_scores)
        self.neg_count += len(neg_scores)

    def compute_epoch_average(self, prefix="train/"):
        """
        バッチごとの集計結果を「バッチ平均」して返す。
        """
        if self.n_batches == 0:
            return {}

        metrics_dict = {
            f"{prefix}accuracy": self.batch_accuracy_sum / self.n_batches,
            f"{prefix}precision": self.batch_precision_sum / self.n_batches,
            f"{prefix}recall": self.batch_recall_sum / self.n_batches,
            f"{prefix}f1": self.batch_f1_sum / self.n_batches,
            f"{prefix}pos_count": self.pos_count / self.n_batches,
            f"{prefix}neg_count": self.neg_count / self.n_batches
        }
        return metrics_dict
