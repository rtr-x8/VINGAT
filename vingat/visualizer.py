import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import pandas as pd
import numpy as np


def visualize_node_pca(data, node_types, title, sample_size=1000):
    """
    HeteroDataを受け取り、指定された複数のnode_typeの特徴量をプロットする関数

    Args:
        data (torch_geometric.data.HeteroData): HeteroDataオブジェクト
        node_types (list): ノードタイプのリスト (e.g., ["user", "item"])
        title (str): プロットのタイトル
        before_training (bool): 訓練前のデータかどうか (デフォルトはTrue)

    Exsample:
        # 使用例
        node_types = ["intention", "taste", "image", "ingredient"]  # ノードタイプのリスト
        # 訓練前のデータで可視化
        visualize_node_pca(train_data, node_types, "Node Feature Visualization")
    """

    all_features = []
    all_labels = []

    for node_type in node_types:
        features = data.x_dict[node_type]  # data.x_dict[node_type]から特徴量を取得

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        num_nodes = features.shape[0]
        if num_nodes > sample_size:
            sampled_indices = np.random.choice(num_nodes, size=sample_size, replace=False)
            features = features[sampled_indices]

        all_features.append(features)
        all_labels.extend([node_type] * len(features))  # ノードタイプをラベルとして追加

    all_features = np.concatenate(all_features)

    # PCAで2次元に次元削減
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_features)

    # DataFrameを作成
    df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df['node_type'] = all_labels

    # Seabornで散布図を作成
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='node_type', data=df, palette='Set1')  # ノードタイプで色分け
    # plt.title(f'{title}')
    plt.show()

    return df


def wandb_pca(wdb, df):
    table = wdb.Table(
        data=df[['PC1', 'PC2', 'node_type']].values,
        columns=['PC1', 'PC2', 'node_type']
    )

    # W&Bに散布図としてログ
    # - `x`, `y`: 散布図の軸
    # - `label`: 色分けしたい列の指定
    # - `title`: タイトルを付けられる
    wdb.log({
        "pca_scatter": wdb.plot.scatter(
            table,
            x="PC1",
            y="PC2",
            label="node_type",
            title="title"
        )
    })


def visualize_node_distribution(train_data, test_data, val_data, n_components=2):
    """
    ノードタイプごとに3列1行のsubplotsを作成し、train_data, test_data, val_dataの特徴量をそれぞれ別のsubplotに表示する関数

    Args:
        train_data (torch_geometric.data.HeteroData): トレーニングデータ
        test_data (torch_geometric.data.HeteroData): テストデータ
        val_data (torch_geometric.data.HeteroData): 検証データ
        n_components (int): PCAで削減する次元数

    Example:
        # 使用例
        visualize_node_distribution(train_data, test_data, val_data)
    """

    def get_node_features(data):
        node_features = {}
        for node_type in data.node_types:
            features = data[node_type].x
            if features is not None:
                node_features[node_type] = features.cpu().numpy()
        return node_features

    train_features = get_node_features(train_data)
    test_features = get_node_features(test_data)
    val_features = get_node_features(val_data)

    # 全てのノードタイプを取得
    tr_keys, test_keys, val_keys = train_features.keys(), test_features.keys(), val_features.keys()
    all_node_types = set(tr_keys) | set(test_keys) | set(val_keys)

    for node_type in all_node_types:
        # 3列1行のsubplotsを作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

        # 各データセットの特徴量をプロット
        for i, (data, label) in enumerate([
            (train_features, "train"),
            (test_features, "test"),
            (val_features, "val")
        ]):
            if node_type in data:
                # PCAで次元削減
                pca = PCA(n_components=n_components)
                reduced_features = pca.fit_transform(data[node_type])

                df = pd.DataFrame(reduced_features)
                df.columns = [f"{node_type}_PC{j+1}" for j in range(n_components)]

                # 散布図で可視化
                sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df, ax=axes[i], label=label)
                axes[i].set_title(f"Distribution of '{node_type}' features ({label})")
                axes[i].set_xlabel(df.columns[0])
                axes[i].set_ylabel(df.columns[1])
                axes[i].legend()  # 凡例を表示

        plt.tight_layout()
        plt.show()
