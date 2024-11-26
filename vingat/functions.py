import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm.notebook import tqdm
import copy
from sklearn.metrics import accuracy_score, recall_score, f1_score
import os
import shutil


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

    with torch.no_grad():
        data = data.to(device)
        out = model(data)

        # 評価用のエッジラベルとエッジインデックスを取得
        edge_label_index = data['user', 'buys', 'recipe'].edge_label_index

        # ユーザーとレシピの埋め込みを取得
        user_embeddings = out['user'].x
        recipe_embeddings = out['recipe'].x

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

            if len(np.unique(labels)) > 1:    # Check if we have both positive and negative samples
                auc = roc_auc_score(labels, scores)
                all_aucs.append(auc)


            # このユーザーのスコアを集計してトップkのレコメンデーションを取得
            user_scores = torch.cat([pos_scores, neg_scores], dim=0).cpu().numpy()
            recipe_indices = np.concatenate([user_pos_indices.cpu().numpy(), user_neg_indices.cpu().numpy()])

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

    return avg_precision, avg_recall, avg_ndcg, avg_accuracy, avg_f1, avg_auc


def save_model(model: nn.Module,  save_directory: str, filename: str):
   os.makedirs(save_directory, exist_ok=True)
   torch.save(model.state_dict(), f"{save_directory}/{filename}.pth")


def train_func(train_loader, val, model, optimizer, criterion, epochs, device, patience=5):
    model.to(device)
    model.train()
    best_val_metric = 0    # 現時点での最良のバリデーションメトリクスを初期化
    patience_counter = 0    # Early Stoppingのカウンターを初期化
    best_model_state = None
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_data in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)

            # モデルのフォワードパス
            out = model(batch_data)

            # エッジのラベルとエッジインデックスを取得
            edge_label = batch_data['user', 'buys', 'recipe'].edge_label
            edge_label_index = batch_data['user', 'buys', 'recipe'].edge_label_index

            # ユーザーとレシピの埋め込みを取得
            user_embeddings = out['user'].x
            recipe_embeddings = out['recipe'].x

            # 正例と負例のマスクを取得
            pos_mask = batch_data['user', 'buys', 'recipe'].edge_label == 1
            neg_mask = batch_data['user', 'buys', 'recipe'].edge_label == 0

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
            loss = criterion(pos_scores, neg_scores, model.parameters())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend((pos_scores > 0.5).int().tolist() + (neg_scores <= 0.5).int().tolist())
            all_labels.extend([1] * len(pos_scores) + [0] * len(neg_scores))

        aveg_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_recall = recall_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        print(f"Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}")

        wandb.log({
            "train/total_loss": total_loss,
            "train/aveg_loss": aveg_loss,
            "train/accuracy": epoch_accuracy,
            "train/recall": epoch_recall,
            "train/f1": epoch_f1
        }, step=epoch+1)

        # Valid
        k = 10
        val_copied = copy.deepcopy(val)
        val_precision, val_recall, val_ndcg, val_accuracy, val_f1, val_auc = evaluate_model(model, val_copied, device, k=k, desc=f"[Valid] Epoch {epoch+1}/{epochs}")

        # 結果を表示
        print(f'Accuracy@{k}: {val_accuracy:.4f}, Recall@{k}: {val_recall:.4f}, F1@{k}: {val_f1:.4f}, Precision@{k}: {val_precision:.4f}, NDCG@{k}: {val_ndcg:.4f}, AUC: {val_auc:.4f}')
        print("===")

        wandb.log({
            f"val/Precision@{k}": val_precision,
            f"val/Recall@{k}": val_recall,
            f"val/NDCG@{k}": val_ndcg,
            f"val/Accuracy@{k}": val_accuracy,
            f"val/F1@{k}": val_f1,
            f"val/AUC": val_auc,
        })

        save_model(model, f"{PATH}/models/{PROJECT_NAME}/{run_name}", f"model_{epoch+1}")

        # Early Stoppingの判定（バリデーションの精度または他のメトリクスで判定）
        if val_accuracy > best_val_metric:
            best_val_metric = val_accuracy    # 最良のバリデーションメトリクスを更新
            patience_counter = 0    # 改善が見られたためカウンターをリセット
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1    # 改善がなければカウンターを増やす

        # patienceを超えた場合にEarly Stoppingを実行
        if patience_counter >= patience:
            print(f"バリデーションメトリクスの改善がないため、エポック{epoch+1}でEarly Stoppingを実行します。")
            wandb.alert(
                title="Early Stopped",
                text=f"学習が終了しました。\nプロジェクト名：{PROJECT_NAME}\n管理番号：{run_name}",
                level=wandb.AlertLevel.ERROR,
            )
            break

    wandb.alert(
        title="訓練終了",
        text=f"学習が終了しました。\nプロジェクト名：{PROJECT_NAME}\n管理番号：{run_name}",
        level=wandb.AlertLevel.ERROR,
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_model(model, f"{PATH}/models/{PROJECT_NAME}/{run_name}", f"best_model")

    return model
