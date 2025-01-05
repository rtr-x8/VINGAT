import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import os
import numpy as np
from torch_geometric.loader import LinkNeighborLoader
from typing import Callable, Dict, List
import pandas as pd
from vingat.metrics import ScoreMetricHandler
from vingat.metrics import MetricsHandler


def evaluate_model(
    model: nn.Module,
    dataloader: LinkNeighborLoader,
    device: torch.device,
    desc: str = ""
):
    model.eval()

    mhandler = MetricsHandler(device=device, threshold=0.5)
    shandler = ScoreMetricHandler(device=device)

    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=desc):
            batch_data = batch_data.to(device)
            out, _ = model(batch_data)

            # 評価用のエッジラベルとエッジインデックスを取得
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

            # スコアの統計量を更新
            shandler.update(pos_scores, neg_scores)
            mhandler.update(
                probas=torch.cat([pos_scores, neg_scores]),
                targets=torch.cat([
                    torch.ones_like(pos_scores, device=device),
                    torch.zeros_like(neg_scores, device=device)
                ]),
                user_indices=torch.cat([
                    edge_label_index[0][pos_mask],
                    edge_label_index[0][neg_mask]
                ])
            )

    shandler.compute()
    mhandler.compute()

    return shandler, mhandler


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
    val_data,
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
):
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    model.to(device)
    best_val_metric = 0    # 現時点での最良のバリデーションメトリクスを初期化
    patience_counter = 0    # Early Stoppingのカウンターを初期化
    best_model_epoch = 0

    save_dir = f"{directory_path}/models/{project_name}/{experiment_name}"

    for epoch in range(1, epochs+1):
        loss_histories: Dict[str, List[torch.Tensor]] = {
            "total_loss": [],
            "main_loss": [],
        }
        node_mean = []
        mhandler = MetricsHandler(device=device, threshold=0.5)

        model.train()

        print(f"Epoch {epoch}/{epochs} ======================")

        for batch_data in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)

            # モデルのフォワードパス
            # out, cl_loss = model(batch_data)
            out, loss_entories = model(batch_data)

            main_loss_rate = 1.0 - sum([entry["weight"] for entry in loss_entories])
            if main_loss_rate < 0:
                raise ValueError("main loss rate is negative")

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
            main_loss = criterion(pos_scores, neg_scores, model.parameters())

            other_loss = torch.sum(torch.stack(
                [entry["loss"] * entry["weight"] for entry in loss_entories]
            ))
            loss = main_loss_rate * main_loss + other_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_histories["total_loss"].append(loss.item())
            loss_histories["main_loss"].append((main_loss_rate * main_loss).item())
            for entry in loss_entories:
                if entry["name"] not in loss_histories.keys():
                    loss_histories[entry["name"]] = []
                loss_histories[entry["name"]].append(
                    (entry["loss"] * entry["weight"]).item()
                )

            mhandler.update(
                probas=torch.cat([pos_scores, neg_scores]),
                targets=torch.cat([
                    torch.ones_like(pos_scores, device=device),
                    torch.zeros_like(neg_scores, device=device)
                ]),
                user_indices=torch.cat([
                    edge_label_index[0][pos_mask],
                    edge_label_index[0][neg_mask]
                ])
            )

            # check
            node_mean.append({
                key: val.mean().mean().item()
                for key, val in out.x_dict.items()
            })

        df = calculate_statistics(node_mean)
        print("Score Statics: ")
        print(df)

        tr_metrics = {
            f"train-loss/{k}": np.mean(v)
            for k, v in loss_histories.items()
        }
        print("Loss: ")
        print(tr_metrics)
        wbLogger(
            data=tr_metrics,
            step=epoch
        )

        wbLogger(data=mhandler.log("train-handler"), step=epoch)
        print("handler Result: ")
        print(mhandler.log(prefix="train-handler", num_round=4))

        # Valid
        if epoch % validation_interval == 0:

            print("Validation -------------------")

            """
            _df = visualize_node_pca(batch_data,
                                     pca_cols,
                                     f"after_training. Epoch: {epoch}/{epochs}")
            wbScatter(_df, epoch, title=f"after training (epoch: {epoch})")
            """

            score_statics, v_mhandler = evaluate_model(
                model, val_data, device, desc=f"[Valid] Epoch {epoch}/{epochs}")

            val_metrics = {
                "val/last_lr": scheduler.get_last_lr()[0]
            }

            wbLogger(
                data=val_metrics,
                step=epoch
            )
            print("Score Statics: ")
            print(score_statics.log(prefix="val-score-statics", num_round=4))
            wbLogger(data=score_statics.log(prefix="val-score-statics"), step=epoch)

            print("handler Result: ")
            vmhlog = v_mhandler.log(prefix="val-handler", num_round=4)
            print(vmhlog)
            wbLogger(data=vmhlog, step=epoch)

            save_model(model, save_dir, f"model_{epoch}")

            v_base_metric = v_mhandler.compute().get("precision@10")

            # Early Stoppingの判定（バリデーションの精度または他のメトリクスで判定）
            if v_base_metric > best_val_metric:
                best_val_metric = v_base_metric    # 最良のバリデーションメトリクスを更新
                patience_counter = 0    # 改善が見られたためカウンターをリセット
                best_model_epoch = epoch
            else:
                patience_counter += 1    # 改善がなければカウンターを増やす

            # patienceを超えた場合にEarly Stoppingを実行
            if patience_counter >= patience:
                print(f"エポック{epoch}でEarly Stoppingを実行します。")
                wbTagger("early_stopped")
                model.load_state_dict(torch.load(f"{save_dir}/model_{best_model_epoch}.pth"))
                break

            print(f"patience_counter: {patience_counter} / {patience}")

        scheduler.step()

    if epochs <= epoch:
        wbTagger("epoch_completed")
    else:
        wbTagger(f"under_{(epoch // 10 + 1) * 10}_epochs")

    return model
