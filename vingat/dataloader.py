from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import torch
from enum import Enum
import numpy as np
from vingat.loader import use_nutritions

import pandas as pd
from torch_geometric.loader import LinkNeighborLoader


class RecipeFeatureType(Enum):
    VISUAL = 0
    INTENTION = 1
    TASTE = 2


def create_hetrodata(
    ratings: pd.DataFrame,
    ingredients: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    recipe_nutrients: pd.DataFrame,
    # cooking_direction_embeddings: torch.Tensor,
    user_label_encoder: LabelEncoder,
    recipe_label_encoder: LabelEncoder,
    ingredient_label_encoder: LabelEncoder,
    device,
    hidden_dim: int = 128
) -> HeteroData:

    hetro = HeteroData()

    # Node #

    # User nodes
    num_users = len(user_label_encoder.classes_)
    user_x = torch.zeros((num_users, hidden_dim), dtype=torch.float32)
    user_id = torch.tensor(user_label_encoder.classes_, dtype=torch.long)
    hetro["user"].x = user_x
    hetro["user"].user_id = user_id
    hetro["user"].num_nodes = num_users

    # Recipe nodes
    num_recipes = len(recipe_label_encoder.classes_)
    recipe_x = torch.zeros((num_recipes, hidden_dim), dtype=torch.float32)
    recipe_id = torch.tensor(recipe_label_encoder.classes_, dtype=torch.long)
    hetro["item"].x = recipe_x
    hetro["item"].recipe_id = recipe_id
    hetro["item"].num_nodes = num_recipes

    # Image nodes (one-to-one with recipes)
    hetro["image"].x = torch.zeros((num_recipes, hidden_dim), dtype=torch.float32)
    hetro["image"].recipe_id = recipe_id
    hetro["image"].num_nodes = num_recipes

    # Intention nodes (one-to-one with recipes)
    nutrient_features = recipe_nutrients.loc[
        recipe_label_encoder.classes_, use_nutritions].values
    hetro["intention"].x = torch.zeros((num_recipes, hidden_dim), dtype=torch.float32)
    hetro["intention"].nutrient = torch.tensor(nutrient_features, dtype=torch.float32)
    hetro["intention"].recipe_id = recipe_id
    hetro["intention"].num_nodes = num_recipes

    # Taste nodes (one-to-one with recipes)
    # hetro["taste"].x = cooking_direction_embeddings.to(torch.float32)
    hetro["taste"].x = torch.zeros((num_recipes, hidden_dim), dtype=torch.float32)
    hetro["taste"].recipe_id = recipe_id
    hetro["taste"].num_nodes = num_recipes

    # Ingredient nodes
    num_ingredients = len(ingredient_label_encoder.classes_)
    ingredient_x = torch.zeros((num_ingredients, hidden_dim), dtype=torch.float32)
    ingredient_id = torch.tensor(ingredient_label_encoder.classes_, dtype=torch.long)
    hetro["ingredient"].x = ingredient_x
    hetro["ingredient"].ingredient_id = ingredient_id
    hetro["ingredient"].num_nodes = num_ingredients

    # Edge #

    # Edges between user and recipe (buys)
    edge_index_user_recipe = torch.tensor([
        user_label_encoder.transform(ratings["user_id"].values),
        recipe_label_encoder.transform(ratings["recipe_id"].values)
    ], dtype=torch.long)
    hetro["user", "buys", "item"].edge_index = edge_index_user_recipe
    hetro["item", "bought_by", "user"].edge_index = edge_index_user_recipe.flip(0)

    # Edges between image and recipe (one-to-one)
    edge_index_image_recipe = torch.stack([
        torch.arange(num_recipes),
        torch.arange(num_recipes)
    ], dim=0)
    hetro["image", "associated_with", "item"].edge_index = edge_index_image_recipe
    hetro["item", "has_image", "image"].edge_index = edge_index_image_recipe.flip(0)

    # Edges between intention and recipe (one-to-one)
    edge_index_intention_recipe = torch.stack([
        torch.arange(num_recipes),
        torch.arange(num_recipes)
    ], dim=0)
    hetro["intention", "associated_with", "item"].edge_index = edge_index_intention_recipe
    hetro["item", "has_intention", "intention"].edge_index = edge_index_intention_recipe.flip(0)

    # Edges between taste and recipe (one-to-one)
    edge_index_taste_recipe = torch.stack([
        torch.arange(num_recipes),
        torch.arange(num_recipes)
    ], dim=0)
    hetro["taste", "associated_with", "item"].edge_index = edge_index_taste_recipe
    hetro["item", "has_taste", "taste"].edge_index = edge_index_taste_recipe.flip(0)

    # ing to taste(recupe)
    edge_index_ingredient_recipe = np.array([
        ingredient_label_encoder.transform(recipe_ingredients["ingredient_id"]),
        recipe_label_encoder.transform(recipe_ingredients["recipe_id"])])
    edge_index_taste_ingredient = torch.tensor(edge_index_ingredient_recipe, dtype=torch.long)
    hetro["taste", "contains", "ingredient"].edge_index = edge_index_taste_ingredient
    hetro["ingredient", "part_of", "taste"].edge_index = edge_index_taste_ingredient.flip(0)

    # for LinkNeighborLoader #
    hetro['user', 'buys', 'item'].edge_label = torch.ones(
        edge_index_user_recipe.shape[1],
        dtype=torch.long)
    hetro["user", "buys", "item"].edge_label_index = torch.tensor(
        edge_index_user_recipe, dtype=torch.long)

    # edge_index の包括的なチェック
    # ユーザーインデックスのチェック
    if hetro["user"].x.shape[0] >= num_users:
        print(hetro["user"].x.shape[0], num_users)
        raise ValueError(f"edge_index のユーザーインデックスがノード数を超えています。")
    if hetro["user"].x.shape[0] < 0:
        print(hetro["user"].x.shape[0])
        raise ValueError("edge_index のユーザーインデックスに負の値が含まれています。")
    # レシピインデックスのチェック
    if hetro["item"].max() >= num_recipes:
        print(hetro["item"].max(), num_recipes)
        raise ValueError("edge_index のレシピインデックスがノード数を超えています。")
    if hetro["item"].min() < 0:
        print(hetro["item"].min())
        raise ValueError("edge_index のレシピインデックスに負の値が含まれています。")

    hetro.to(device)
    return hetro


def create_data(
    core_train_rating: pd.DataFrame,
    core_test_rating: pd.DataFrame,
    core_val_rating: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    ingredients: pd.DataFrame,
    recipe_nutrients: pd.DataFrame,
    device: torch.device,
    hidden_dim: int = 128
):
    all_user_id = pd.concat([
        core_train_rating["user_id"],
        core_test_rating["user_id"],
        core_val_rating["user_id"]
    ]).unique()

    all_recipe_id = pd.concat([
        core_train_rating["recipe_id"],
        core_test_rating["recipe_id"],
        core_val_rating["recipe_id"],
        recipe_ingredients["recipe_id"]
    ]).unique()

    user_label_encoder = LabelEncoder().fit(all_user_id)
    recipe_label_encoder = LabelEncoder().fit(all_recipe_id)
    ingredient_indices = recipe_ingredients["ingredient_id"].unique()
    ingredient_label_encoder = LabelEncoder().fit(ingredient_indices)

    train_recipe_ingedient = recipe_ingredients[
        recipe_ingredients["recipe_id"].isin(core_train_rating["recipe_id"])]
    test_recipe_ingedient = recipe_ingredients[
        recipe_ingredients["recipe_id"].isin(core_test_rating["recipe_id"])]
    val_recipe_ingedient = recipe_ingredients[
        recipe_ingredients["recipe_id"].isin(core_val_rating["recipe_id"])]

    # データの整合性を確認
    def check_data_integrity(df, label_encoder, column_name):
        encoded_values = label_encoder.transform(df[column_name])
        if (encoded_values < 0).any():
            raise ValueError(f"{column_name} に負のインデックスが含まれています。")
        if encoded_values.max() >= len(label_encoder.classes_):
            raise ValueError(f"{column_name} のインデックスがノード数を超えています。")

    check_data_integrity(core_train_rating, user_label_encoder, "user_id")
    check_data_integrity(core_train_rating, recipe_label_encoder, "recipe_id")
    check_data_integrity(core_test_rating, user_label_encoder, "user_id")
    check_data_integrity(core_test_rating, recipe_label_encoder, "recipe_id")
    check_data_integrity(core_val_rating, user_label_encoder, "user_id")
    check_data_integrity(core_val_rating, recipe_label_encoder, "recipe_id")

    train = create_hetrodata(
        core_train_rating, ingredients.copy(), train_recipe_ingedient.copy(),
        recipe_nutrients, user_label_encoder, recipe_label_encoder,
        ingredient_label_encoder, device, hidden_dim)
    test = create_hetrodata(
        core_test_rating, ingredients.copy(), test_recipe_ingedient.copy(),
        recipe_nutrients, user_label_encoder, recipe_label_encoder,
        ingredient_label_encoder, device, hidden_dim)
    val = create_hetrodata(
        core_val_rating, ingredients.copy(), val_recipe_ingedient.copy(),
        recipe_nutrients, user_label_encoder, recipe_label_encoder,
        ingredient_label_encoder, device, hidden_dim)

    return (
        train,
        test,
        val,
        user_label_encoder,
        recipe_label_encoder,
        ingredient_label_encoder
    )


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
            ('user', 'buys', 'item'): [10, 5],
            ('item', 'bought_by', 'user'): [10, 5],
            ('image', 'associated_with', 'item'): [1, 0],
            ('intention', 'associated_with', 'item'): [1, 0],
            ('taste', 'associated_with', 'item'): [1, 0],
            ('taste', 'contains', 'ingredient'): [10, 5],
            ('ingredient', 'part_of', 'taste'): [10, 5],
            ('item', 'has_image', 'image'): [1, 0],
            ('item', 'has_intention', 'intention'): [1, 0],
            ('item', 'has_taste', 'taste'): [1, 0],
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
