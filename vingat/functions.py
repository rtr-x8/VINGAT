import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import os
import numpy as np
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from vingat.metrics import ndcg_at_k
from typing import Callable
import pandas as pd
from vingat.visualizer import visualize_node_pca
from vingat.metrics import score_stastics


def evaluate_model(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    k=10,
    desc: str = ""
):
    model.eval()
    all_recalls = []
    all_precisions = []
    all_ndcgs = []
    all_accuracies = []
    all_f1_scores = []
    all_aucs = []
    user_pos_scores = []
    user_neg_scores = []

    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    with torch.no_grad():
        data = data.to(device)
        # out, _ = model(data)
        out = model(data)

        # 評価用のエッジラベルとエッジインデックスを取得
        edge_label_index = data['user', 'buys', 'item'].edge_label_index

        # ユーザーとレシピの埋め込みを取得
        user_embeddings = out['user'].x
        recipe_embeddings = out['item'].x

        # userのindexを取得
        unique_user_ids = edge_label_index[0].unique()

        # 各ユーザーごとにループ
        for user_id in tqdm(unique_user_ids.cpu().numpy(), desc=desc):
            # 現在のユーザーに対する正例ペアを取得
            user_pos_indices = edge_label_index[1][edge_label_index[0] == user_id]
            num_pos_samples = len(user_pos_indices)    # 正例ペアの数を取得
            if num_pos_samples == 0:
                continue

            # 既存エッジのセットを生成
            user_edge_label_index = torch.stack([
                torch.full((num_pos_samples,), user_id, dtype=torch.long, device=device),
                user_pos_indices
            ])

            num_neg_samples = 500    # 負例ペアの数 HAFR, HCGAN
            # PyTorch Geometricのnegative_samplingを使用して負例を取得
            negative_edge_index = negative_sampling(
                edge_index=user_edge_label_index,
                num_nodes=(1, recipe_embeddings.shape[0]),    # (ユーザーのノード数, レシピのノード数)
                num_neg_samples=num_neg_samples,
                force_undirected=False
            )

            # 負例のインデックスを取得
            user_neg_indices = negative_edge_index[1]

            # 正例と負例のユーザーとレシピの埋め込みを作成
            pos_user_embed = user_embeddings[user_id].expand(num_pos_samples, -1)
            pos_recipe_embed = recipe_embeddings[user_pos_indices]

            neg_user_embed = user_embeddings[user_id].expand(num_neg_samples, -1)
            neg_recipe_embed = recipe_embeddings[user_neg_indices]

            # 正例と負例のスコアを計算
            pos_scores = model.predict(pos_user_embed, pos_recipe_embed).squeeze(dim=1)
            neg_scores = model.predict(neg_user_embed, neg_recipe_embed).squeeze(dim=1)

            scores = torch.cat([pos_scores, neg_scores], dim=0).cpu().numpy()
            labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

            user_pos_scores.append(pos_scores)
            user_neg_scores.append(neg_scores)

            if len(np.unique(labels)) > 1:    # Check if we have both positive and negative samples
                auc = roc_auc_score(labels, scores)
                all_aucs.append(auc)

            # このユーザーのスコアを集計してトップkのレコメンデーションを取得
            user_scores = torch.cat([pos_scores, neg_scores], dim=0).cpu().numpy()
            recipe_indices = np.concatenate([
                user_pos_indices.cpu().numpy(),
                user_neg_indices.cpu().numpy()
            ])

            k = min(k, len(user_scores))
            sorted_indices = np.argsort(-user_scores)    # 降順にソート
            top_k_indices = recipe_indices[sorted_indices[:k]]

            # 実際の購入レシピと評価指標の計算
            test_purchased = user_pos_indices.cpu().numpy()
            if len(test_purchased) > 0:
                hits = np.isin(top_k_indices, test_purchased).astype(np.float32)
                recall = hits.sum() / len(test_purchased)
                precision = hits.sum() / k
                ndcg = ndcg_at_k(hits, k)
                accuracy = hits.sum() / k
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                all_recalls.append(recall)
                all_precisions.append(precision)
                all_ndcgs.append(ndcg)
                all_accuracies.append(accuracy)
                all_f1_scores.append(f1)

    avg_recall = np.mean(all_recalls)
    avg_precision = np.mean(all_precisions)
    avg_ndcg = np.mean(all_ndcgs)
    avg_accuracy = np.mean(all_accuracies)
    avg_f1 = np.mean(all_f1_scores)
    avg_auc = np.mean(all_aucs) if all_aucs else 0.0
    score_statics = score_stastics(user_pos_scores, user_neg_scores)

    return avg_precision, avg_recall, avg_ndcg, avg_accuracy, avg_f1, avg_auc, score_statics


def save_model(model: nn.Module,  save_directory: str, filename: str):
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), f"{save_directory}/{filename}.pth")


def calculate_statistics(data):
    """
    与えられた形式のデータを、項目ごとに最小、最大、平均、標準偏差を出算したDataFrameに変換する関数

    Args:
        data (list): 辞書のリスト形式のデータ

    Returns:
        pandas.DataFrame: 統計量をまとめたDataFrame
    """

    # 項目名を取得
    items = list(data[0].keys())

    # 統計量を格納する辞書
    statistics = {}
    for item in items:
        values = [d[item] for d in data]
        statistics[item] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
        }

    # DataFrameに変換
    df = pd.DataFrame(statistics).T

    return df


def train_func(
    train_loader,
    val,
    model,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device,
    wbLogger: Callable,
    wbTagger: Callable,
    wbScatter: Callable,
    directory_path: str,
    project_name: str,
    experiment_name: str,
    patience=4,
    validation_interval=5,
    max_grad_norm=1.0,
    pca_cols=["user", "item", "intention", "taste", "image"],
    cl_loss_rate=0.3
):
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    model.to(device)
    best_val_metric = 0    # 現時点での最良のバリデーションメトリクスを初期化
    patience_counter = 0    # Early Stoppingのカウンターを初期化

    save_dir = f"{directory_path}/models/{project_name}/{experiment_name}"

    for epoch in range(start=1, stop=epochs):
        total_loss = 0
        all_preds = []
        all_labels = []

        model.train()

        node_mean = []

        for batch_data in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)

            # モデルのフォワードパス
            # out, cl_loss = model(batch_data)
            out = model(batch_data)

            # エッジのラベルとエッジインデックスを取得
            # edge_label = batch_data['user', 'buys', 'item'].edge_label
            edge_label_index = batch_data['user', 'buys', 'item'].edge_label_index

            # ユーザーとレシピの埋め込みを取得
            user_embeddings = out['user'].x
            recipe_embeddings = out['item'].x

            # 正例と負例のマスクを取得
            pos_mask = batch_data['user', 'buys', 'item'].edge_label == 1
            neg_mask = batch_data['user', 'buys', 'item'].edge_label == 0

            # エッジインデックスからノードの埋め込みを取得
            user_embed = user_embeddings[edge_label_index[0]]
            recipe_embed = recipe_embeddings[edge_label_index[1]]

            # 正例のスコアを計算
            pos_user_embed = user_embed[pos_mask]
            pos_recipe_embed = recipe_embed[pos_mask]
            pos_scores = model.predict(pos_user_embed, pos_recipe_embed).squeeze()

            # 負例のスコアを計算
            neg_user_embed = user_embed[neg_mask]
            neg_recipe_embed = recipe_embed[neg_mask]
            neg_scores = model.predict(neg_user_embed, neg_recipe_embed).squeeze()

            # 損失の計算
            bpr_loss = criterion(pos_scores, neg_scores, model.parameters())
            # rated_bpr_loss = (1 - cl_loss_rate) * bpr_loss
            # rated_cl_loss = cl_loss_rate * cl_loss

            # loss = rated_bpr_loss + cl_loss_rate * rated_cl_loss

            bpr_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += bpr_loss.item()
            all_preds.extend((pos_scores > 0.5).int().tolist() + (neg_scores <= 0.5).int().tolist())
            all_labels.extend([1] * len(pos_scores) + [0] * len(neg_scores))

            # check
            node_mean.append({
                key: val.mean().mean().item()
                for key, val in out.x_dict.items()
            })
        # print("bpr_loss: ", bpr_loss)
        # print("cl_loss: ", cl_loss, ", loss: ", loss)
        # print("rated_bpr_loss: ", rated_bpr_loss, ", rated_cl_loss: ", rated_cl_loss)

        df = calculate_statistics(node_mean)
        print(df)

        aveg_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_recall = recall_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds)
        epoch_pre = precision_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        txt = f"Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f},"
        txt = f"{txt}, {scheduler.get_last_lr()}"
        print(f"{epoch}/{epochs}", f"{txt} Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}")

        wbLogger(
            data={
                "train/total_loss": total_loss,
                "train/aveg_loss": aveg_loss,
                "train/accuracy": epoch_accuracy,
                "train/recall": epoch_recall,
                "train/precision": epoch_pre,
                "train/f1": epoch_f1,
            },
            step=epoch
        )

        # Valid
        if epoch % validation_interval == 0:

            _df = visualize_node_pca(batch_data,
                                     pca_cols,
                                     f"after_training. Epoch: {epoch}/{epochs}")
            wbScatter(_df, epoch + 1, title=f"after training (epoch: {epoch})")

            k = 10
            v_precision, v_recall, v_ndcg, v_accuracy, v_f1, v_auc, score_statics = evaluate_model(
                model, val, device, k=k, desc=f"[Valid] Epoch {epoch}/{epochs}")

            # 結果を表示
            txt = f'Acc@{k}: {v_accuracy:.4f}, Recall@{k}: {v_recall:.4f},'
            txt = f"{txt} F1@{k}: {v_f1:.4f}, Pre@{k}: {v_precision:.4f},"
            txt = f"{txt} NDCG@{k}: {v_ndcg:.4f}, AUC: {v_auc:.4f}"
            txt = f"{txt}, {scheduler.get_last_lr()}"
            print(txt)
            print("===")

            wbLogger(
                data={
                    f"val/Precision@{k}": v_precision,
                    f"val/Recall@{k}": v_recall,
                    f"val/NDCG@{k}": v_ndcg,
                    f"val/Accuracy@{k}": v_accuracy,
                    f"val/F1@{k}": v_f1,
                    "val/AUC": v_auc,
                }
            )
            print(score_statics)
            wbLogger(**score_statics)

            save_model(model, save_dir, f"model_{epoch}")

            # Early Stoppingの判定（バリデーションの精度または他のメトリクスで判定）
            if v_accuracy > best_val_metric:
                best_val_metric = v_accuracy    # 最良のバリデーションメトリクスを更新
                patience_counter = 0    # 改善が見られたためカウンターをリセット
            else:
                patience_counter += 1    # 改善がなければカウンターを増やす

            # patienceを超えた場合にEarly Stoppingを実行
            if patience_counter >= patience:
                print(f"エポック{epoch}でEarly Stoppingを実行します。")
                wbTagger("early_stopped")
                break

        scheduler.step()

    if epochs == epoch + 1:
        wbTagger("epoch_completed")
    else:
        wbTagger(f"under_{(epoch // 10 + 1) * 10}_epochs")

    return model
