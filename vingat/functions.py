import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import os
import numpy as np
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
from typing import Callable, Dict, List
import pandas as pd
from vingat.metrics import ScoreMetricHandler
from vingat.metrics import MetricsHandler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pytz


def now():
    return datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')


def evaluate_model(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    desc: str = ""
):
    model.eval()

    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    with torch.no_grad():
        mhandler = MetricsHandler(device=device, threshold=0.5)
        shandler = ScoreMetricHandler(device=device)

        data = data.to(device)
        out, _ = model(data)

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

            # スコアの統計量を更新
            shandler.update(pos_scores, neg_scores)
            mhandler.update(
                probas=torch.cat([pos_scores, neg_scores]),
                targets=torch.cat([
                    torch.ones_like(pos_scores, device=device),
                    torch.zeros_like(neg_scores, device=device)
                ]),
                user_indices=torch.full((len(pos_scores) + len(neg_scores),),
                                        user_id, device=device)
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
    def min_max(x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min+1e-8)
        return result

    # 項目名を取得
    items = list(data[0].keys())

    # 統計量を格納する辞書
    statistics = {}
    for item in items:
        values = [d[item] for d in data]
        values_minmax = min_max(np.array(values))
        hist, bins = np.histogram(values_minmax,
                                  range=(0, 1),
                                  bins=np.linspace(0, 1, 6))
        statistics[item] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values),
            **{f"bin_{round(v, 1)}_rate": h/len(values) for h, v in zip(hist, bins)}
        }

    # DataFrameに変換
    df = pd.DataFrame(statistics).T

    return df


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    max_grad_norm: float,
):
    model.to(device)

    loss_histories: Dict[str, List[torch.Tensor]] = {
        "total_loss": [],
        "main_loss": [],
    }
    node_mean = []
    mhandler = MetricsHandler(device=device, threshold=0.5)
    shandler = ScoreMetricHandler(device=device)

    model.train()

    for batch_data in tqdm(train_loader, desc="[Train]"):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)

        out, loss_entories = model(batch_data)

        main_loss_rate = 1.0
        if len(loss_entories) > 0:
            main_loss_rate -= sum([entry["weight"] for entry in loss_entories])

        if main_loss_rate < 0:
            raise ValueError("main loss rate is negative")

        # エッジのラベルとエッジインデックスを取得
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

        loss = main_loss_rate * main_loss
        if len(loss_entories) > 0:
            other_loss = torch.sum(torch.stack(
                [entry["loss"] * entry["weight"] for entry in loss_entories]
            ))
            loss += other_loss

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
        shandler.update(pos_scores, neg_scores)

        # check
        node_mean.append({
            key: val.mean().mean().item()
            for key, val in out.x_dict.items()
        })

    node_stats = calculate_statistics(node_mean)

    return (
        model,
        loss_histories,
        node_stats,
        mhandler,
        shandler,
    )


def show_model_parameters(model: nn.Module, threshold=0.2):
    not_null_norms = []
    not_null_params = []
    null_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            null_params.append(name)
        else:
            not_null_params.append(name)
            not_null_norms.append(param.grad.norm().item())
    average = sum(not_null_norms) / len(not_null_norms)
    upper_threshold = average * (1 + threshold)
    lower_threshold = average * (1 - threshold)

    print(f"Parameter Count: {len(not_null_params) + len(null_params)}")

    if len(not_null_params) > 0:
        print("Parameters Describe: ")
        print(pd.DataFrame(not_null_norms).describe().T)

        warnings = [
            f"[{not_null_param}] is out of average range: {not_null_norm}"
            for not_null_param, not_null_norm in zip(not_null_params, not_null_norms)
            if not (lower_threshold <= not_null_norm <= upper_threshold)
        ]
        if len(warnings) > 0:
            print(f"Warning Parameters: {lower_threshold} <= x <= {upper_threshold}")
            for warning in warnings:
                print(warning)

    if len(null_params) > 0:
        print("Null Parameters: ")
        for null_param in null_params:
            print(f"[{null_param}] is Null.")


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

    best_val_metric = 0    # 現時点での最良のバリデーションメトリクスを初期化
    patience_counter = 0    # Early Stoppingのカウンターを初期化
    best_model_epoch = 0

    save_dir = f"{directory_path}/models/{project_name}/{experiment_name}"

    for epoch in range(1, epochs+1):

        print("\n======================\n", f"Epoch {epoch}/{epochs}", now())
        model, loss_histories, node_stats, mhandler, shandler = train_one_epoch(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            max_grad_norm=max_grad_norm
        )

        print("[Train] Node Statics: ")
        print(node_stats)

        tr_loss = {
            f"train-loss/{k}": np.mean(v)
            for k, v in loss_histories.items()
        }
        print("\n[Train] Loss: ", "\n", tr_loss)
        wbLogger(
            data=tr_loss,
            step=epoch
        )

        print("\n[Train] Metrics: ")
        print(mhandler.log(prefix="train-handler", num_round=4))
        wbLogger(data=mhandler.log("train-handler"), step=epoch)

        print("\n[Train] Score Statics: ")
        print(shandler.log(prefix="train-score-statics", num_round=4))
        wbLogger(data=shandler.log(prefix="train-score-statics"), step=epoch)

        print("\n[Train] Model Parameters: ")
        show_model_parameters(model)

        # Valid
        if epoch % validation_interval == 0:

            print("\nValidation -------------------")

            """
            _df = visualize_node_pca(batch_data,
                                     pca_cols,
                                     f"after_training. Epoch: {epoch}/{epochs}")
            wbScatter(_df, epoch, title=f"after training (epoch: {epoch})")
            """

            score_statics, v_mhandler = evaluate_model(
                model=model,
                data=val_data,
                device=device,
                desc=f"[Valid] Epoch {epoch}/{epochs}"
            )

            val_metrics = {
                "val/last_lr": scheduler.get_last_lr()[0]
            }

            wbLogger(
                data=val_metrics,
                step=epoch
            )
            print(now(), "Score Statics: ")
            print(score_statics.log(prefix="val-score-statics", num_round=4))
            wbLogger(data=score_statics.log(prefix="val-score-statics"), step=epoch)

            print(now(), "handler Result: ")
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
                print(now(), f"エポック{epoch}でEarly Stoppingを実行します。")
                wbTagger("early_stopped")
                model.load_state_dict(torch.load(f"{save_dir}/model_{best_model_epoch}.pth"))
                break

            print(now(), f"patience_counter: {patience_counter} / {patience}")

        scheduler.step()

    if epochs <= epoch:
        wbTagger("epoch_completed")
    else:
        wbTagger(f"under_{(epoch // 10 + 1) * 10}_epochs")

    return model


def get_popularity(
    all_rating: pd.DataFrame,
    device: torch.device,
    item_lencoder: LabelEncoder
) -> torch.Tensor:
    res = all_rating.groupby("recipe_id").sum()
    res["idx"] = item_lencoder.transform(res.index)
    res = res[["idx", "rating"]].reset_index(drop=True).set_index("idx")
    return torch.tensor(res.rating.to_numpy()).to(device)


def get_available_indices(
    rating: pd.DataFrame,
    item_lencoder: LabelEncoder,
    device: torch.device
) -> torch.Tensor:
    recipe_ids = item_lencoder.transform(rating["recipe_id"].unique())
    indeices = torch.tensor(recipe_ids).to(device)
    mask = torch.zeros(item_lencoder.classes_.shape[0], dtype=torch.bool).to(device)
    mask[indeices] = True
    return mask


def negative_sampling_by_popularity(
    positive_edge_index: torch.Tensor,
    popularity: torch.Tensor,
    user_index: int,
    availabel_item_mask: torch.Tensor,
    num_neg_samples: int,
    device: torch.device
) -> torch.Tensor:
    positive_edge_index = positive_edge_index[:, positive_edge_index[0] == user_index]
    positive_item_indices = positive_edge_index[1]
    popularity[positive_item_indices] = 0
    popularity[~availabel_item_mask] = 0
    popularity_sum = popularity.sum()
    proba = popularity / popularity_sum
    sample = torch.multinomial(proba, num_neg_samples, replacement=False)
    return torch.stack([
        torch.full((num_neg_samples,), user_index, dtype=torch.long),
        sample
    ]).to(device)
