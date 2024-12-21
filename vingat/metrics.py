import numpy as np
import torch
from typing import List


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
