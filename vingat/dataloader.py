from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from vingat.loader import use_nutritions
import pandas as pd
from torch_geometric.loader import LinkNeighborLoader
from vingat.encoder import StaticEmbeddingLoader
from typing import Tuple, Optional
import copy
from vingat.preprocess import ScalarPreprocess


def create_dataloader(
    data,
    batch_size,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_workers=0
):
    return LinkNeighborLoader(
        data=data,
        num_neighbors={
            ('user', 'buys', 'item'): [20, 10],
            ('item', 'bought_by', 'user'): [20, 10],
            ('image', 'associated_with', 'item'): [20, 10],
            ('intention', 'associated_with', 'item'): [20, 10],
            ('taste', 'associated_with', 'item'): [20, 10],
            ('taste', 'contains', 'ingredient'): [20, 10],
            ('ingredient', 'part_of', 'taste'): [20, 10],
            ('item', 'has_image', 'image'): [20, 10],
            ('item', 'has_intention', 'intention'): [20, 10],
            ('item', 'has_taste', 'taste'): [20, 10],
        },
        edge_label_index=(
            ('user', 'buys', 'item'),
            data['user', 'buys', 'item'].edge_label_index
        ),
        edge_label=data['user', 'buys', 'item'].edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        neg_sampling_ratio=neg_sampling_ratio,
        num_workers=num_workers
    )


def create_base_hetero(
    core_train_rating: pd.DataFrame,
    core_test_rating: pd.DataFrame,
    core_val_rating: pd.DataFrame,
    ingredients: pd.DataFrame,
    recipe_nutrients: pd.DataFrame,
    recipe_image_embeddings: pd.DataFrame,
    recipe_image_vlm_caption_embeddings: pd.DataFrame,
    recipe_cooking_directions_embeddings: pd.DataFrame,
    ingredients_with_embeddings: pd.DataFrame,
    directory_path: str,
    device: torch.device,
    hidden_dim: int
) -> Tuple[HeteroData, LabelEncoder, LabelEncoder, LabelEncoder]:

    # 全データ
    all_user_ids = pd.concat([
        core_train_rating["user_id"],
        core_test_rating["user_id"],
        core_val_rating["user_id"]]).unique()
    all_item_ids = pd.concat([
        core_train_rating["recipe_id"],
        core_test_rating["recipe_id"],
        core_val_rating["recipe_id"]]).unique()
    all_ingredient_ids = ingredients.index.to_list()

    user_lencoder = LabelEncoder().fit(all_user_ids)
    item_lencoder = LabelEncoder().fit(all_item_ids)
    ing_lencoder = LabelEncoder().fit(all_ingredient_ids)

    _df = recipe_nutrients.copy()[use_nutritions]
    _recipe_nutrients = (_df - _df.mean()) / _df.std()

    # hetero
    data = HeteroData()

    # Node
    num_users = len(user_lencoder.classes_)
    data["user"].num_nodes = num_users
    data["user"].user_id = torch.tensor(user_lencoder.classes_)
    data["user"].x = torch.zeros((num_users, hidden_dim), dtype=torch.float32)
    data["user"].id = torch.range(0, num_users - 1, dtype=torch.long).unsqueeze(1)

    num_items = len(item_lencoder.classes_)
    data["item"].num_nodes = num_items
    data["item"].item_id = torch.tensor(item_lencoder.classes_)
    data['item'].x = torch.zeros((num_items, hidden_dim), dtype=torch.float32)
    data["item"].id = torch.range(0, num_items - 1, dtype=torch.long).unsqueeze(1)

    data["image"].num_nodes = len(item_lencoder.classes_)
    data["image"].item_id = torch.tensor(item_lencoder.classes_)
    image_encoder = StaticEmbeddingLoader(recipe_image_embeddings,
                                          dimention=hidden_dim,
                                          device=device)
    data["image"].x = image_encoder(torch.tensor(item_lencoder.classes_, dtype=torch.long))

    data["intention"].num_nodes = len(item_lencoder.classes_)
    data["intention"].item_id = torch.tensor(item_lencoder.classes_)
    data["intention"].nutrient = torch.tensor(
        _recipe_nutrients.loc[item_lencoder.classes_, use_nutritions].values,
        dtype=torch.float32)
    caption_encoder = StaticEmbeddingLoader(recipe_image_vlm_caption_embeddings,
                                            dimention=hidden_dim,
                                            device=device)
    data["intention"].x = caption_encoder(torch.tensor(item_lencoder.classes_, dtype=torch.long))

    data["taste"].num_nodes = len(item_lencoder.classes_)
    data["taste"].item_id = torch.tensor(item_lencoder.classes_)
    vlm_encoder = StaticEmbeddingLoader(
        recipe_cooking_directions_embeddings,
        dimention=hidden_dim, device=device)
    data["taste"].x = vlm_encoder(torch.tensor(item_lencoder.classes_, dtype=torch.long))

    data["ingredient"].num_nodes = len(ing_lencoder.classes_)
    data["ingredient"].ingredient_id = torch.tensor(ing_lencoder.classes_)
    ingre_encoder = StaticEmbeddingLoader(ingredients_with_embeddings,
                                          dimention=hidden_dim,
                                          device=device)
    data["ingredient"].x = ingre_encoder(torch.tensor(ing_lencoder.classes_, dtype=torch.long))

    # Edge
    ei_attr_item = torch.stack([
        torch.arange(len(item_lencoder.classes_)),
        torch.arange(len(item_lencoder.classes_))
    ], dim=0)
    data["image", "associated_with", "item"].edge_index = ei_attr_item.detach().clone()
    data["item", "has_image", "image"].edge_index = ei_attr_item.detach().clone().flip(0)
    data["intention", "associated_with", "item"].edge_index = ei_attr_item.detach().clone()
    data["item", "has_intention", "intention"].edge_index = ei_attr_item.detach().clone().flip(0)
    data["taste", "associated_with", "item"].edge_index = ei_attr_item.detach().clone()
    data["item", "has_taste", "taste"].edge_index = ei_attr_item.detach().clone().flip(0)

    data.to(device=device)

    return data, user_lencoder, item_lencoder, ing_lencoder


def mask_hetero(
    hetero: HeteroData,
    rating: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    user_lencoder: LabelEncoder,
    item_lencoder: LabelEncoder,
    ing_lencoder: LabelEncoder,
    is_train: bool,
    scalar_preprocess: Optional[ScalarPreprocess] = None,
) -> Tuple[HeteroData, ScalarPreprocess]:

    # 環境ごとのデータ
    user_recipe_set = rating[["user_id", "recipe_id"]]
    ing_item = recipe_ingredients.loc[
        recipe_ingredients["recipe_id"].isin(user_recipe_set["recipe_id"])
    ]

    data = copy.deepcopy(hetero)

    # ユーザーIDのセット差分を取得（全体のクラスから現在のデータセットに存在するIDを引く）
    no_user_ids = np.setdiff1d(user_lencoder.classes_, user_recipe_set["user_id"].values)
    no_item_ids = np.setdiff1d(item_lencoder.classes_, user_recipe_set["recipe_id"].values)

    # 存在しないユーザーIDおよびアイテムIDをエンコード（インデックスに変換）
    # LabelEncoderは既に全体のIDに対してフィットされているため、エラーは発生しない
    no_user_indices = user_lencoder.transform(no_user_ids)
    no_item_indices = item_lencoder.transform(no_item_ids)

    data.x_dict["user"][no_user_indices] = data.x_dict["user"][no_user_indices].zero_()
    data.x_dict["item"][no_item_indices] = data.x_dict["item"][no_item_indices].zero_()

    # 標準化
    if scalar_preprocess is None:
        if is_train:
            scalar_preprocess = ScalarPreprocess(data.x_dict)
            scalar_preprocess.fit()
        else:
            raise ValueError("scalar_preprocess must be provided when is_train is False.")
    data.x_dict.update(scalar_preprocess.transform(data.x_dict))

    # edge
    edge_index_user_recipe = torch.tensor([
        user_lencoder.transform(rating["user_id"].values),
        item_lencoder.transform(rating["recipe_id"].values)
    ], dtype=torch.long)
    data["user", "buys", "item"].edge_index = edge_index_user_recipe
    data["item", "bought_by", "user"].edge_index = edge_index_user_recipe.detach().clone().flip(0)
    data['user', 'buys', 'item'].edge_label = torch.ones(
        edge_index_user_recipe.shape[1],
        dtype=torch.long)
    data["user", "buys", "item"].edge_label_index = edge_index_user_recipe.clone().detach()

    ei_ing_item = torch.tensor([
        ing_lencoder.transform(ing_item["ingredient_id"].values),
        item_lencoder.transform(ing_item["recipe_id"].values)
    ], dtype=torch.long)
    data["ingredient", "part_of", "taste"].edge_index = ei_ing_item
    data["taste", "contains", "ingredient"].edge_index = ei_ing_item.detach().clone().flip(0)

    return data, scalar_preprocess
