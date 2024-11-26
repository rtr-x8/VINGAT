from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import negative_sampling
import torch
import torch.nn as nn
from enum import Enum
import numpy as np
from .loader import use_nutritions, load_recipe_nutrients

class RecipeFeatureType(Enum):
    VISUAL = 0
    INTENTION = 1
    TASTE = 2


def create_hetrodata(ratings: pd.DataFrame,
                     ingredients: pd.DataFrame,
                     recipe_ingredients: pd.DataFrame,
                     user_label_encoder: LabelEncoder,
                     recipe_label_encoder: LabelEncoder,
                     ingredient_label_encoder: LabelEncoder,
                     device,
                     recipe_nutrients) -> HeteroData:

    hetro = HeteroData()

    user_features = user_label_encoder.classes_ # 全てのユーザーノードを登録する必要がる
    user_indices = np.arange(0, len(user_label_encoder.classes_))
    hetro["user"].x = torch.tensor(user_features, dtype=torch.float32).unsqueeze(1)
    hetro["user"].id = torch.tensor(user_indices, dtype=torch.long) # 連番に振り直す
    hetro["user"].num_nodes = len(user_features)

    recipe_features = recipe_nutrients.loc[recipe_label_encoder.classes_, use_nutritions].values
    recipe_indices = np.arange(0, len(recipe_label_encoder.classes_))
    hetro["recipe"].x = torch.tensor(recipe_features, dtype=torch.float32)
    hetro["recipe"].id = torch.tensor(recipe_label_encoder.classes_, dtype=torch.long)
    hetro["recipe"].num_nodes = len(recipe_label_encoder.classes_)
    hetro["recipe"].visual_feature = torch.ones((len(recipe_label_encoder.classes_), 1), dtype=torch.float32)
    hetro["recipe"].intention_feature = torch.ones((len(recipe_label_encoder.classes_), 1), dtype=torch.float32)
    hetro["recipe"].taste_feature = torch.ones((len(recipe_label_encoder.classes_), 1), dtype=torch.float32)

    ingredient_features = ingredient_label_encoder.classes_
    ingredient_indices = np.arange(0, len(ingredient_label_encoder.classes_))
    hetro["ingredient"].x = torch.tensor(ingredient_features, dtype=torch.float32)
    hetro["ingredient"].id = torch.tensor(ingredient_label_encoder.classes_, dtype=torch.long)
    hetro["ingredient"].num_nodes = len(ingredient_label_encoder.classes_)

    # edgeはデータに基づくリンクだけ設定する。
    edge_index_user_recipe = np.array([user_label_encoder.transform(ratings["user_id"]), recipe_label_encoder.transform(ratings["recipe_id"])])
    hetro["user", "buys", "recipe"].edge_index = torch.tensor(edge_index_user_recipe, dtype=torch.long)
    hetro["recipe", "bought_by", "user"].edge_index = torch.tensor(edge_index_user_recipe, dtype=torch.long).flip(0)

    # 自己ループ
    self_loop_edges = torch.arange(len(recipe_label_encoder.classes_)).unsqueeze(0).repeat(2, 1)
    hetro['recipe', 'self_loop', 'recipe'].edge_index = self_loop_edges

    # for LinkNeighborLoader
    hetro['user', 'buys', 'recipe'].edge_label = torch.ones(edge_index_user_recipe.shape[1], dtype=torch.long)
    hetro["user", "buys", "recipe"].edge_label_index = torch.tensor(edge_index_user_recipe, dtype=torch.long)

    edge_index_ingredient_recipe = np.array([ingredient_label_encoder.transform(recipe_ingredients["ingredient_id"]),
                                                recipe_label_encoder.transform(recipe_ingredients["recipe_id"])])
    hetro["ingredient", "used_in", "recipe"].edge_index = torch.tensor(edge_index_ingredient_recipe, dtype=torch.long)

    hetro.to(device)

    return hetro

def create_dataloader(
    core_train_rating: pd.DataFrame,
    core_test_rating: pd.DataFrame,
    core_val_rating: pd.DataFrame,
    recipe_ingredients: pd.DataFrame,
    ingredients: pd.DataFrame,
    create_hetrodata: callable,
    device: torch.device
):
    all_user_id = pd.concat([core_train_rating["user_id"], core_test_rating["user_id"], core_val_rating["user_id"]]).unique()
    all_recipe_id = pd.concat([core_train_rating["recipe_id"], core_test_rating["recipe_id"], core_val_rating["recipe_id"], recipe_ingredients["recipe_id"]]).unique()

    user_label_encoder = LabelEncoder().fit(all_user_id)
    recipe_label_encoder = LabelEncoder().fit(all_recipe_id)
    ingredient_label_encoder = LabelEncoder().fit(recipe_ingredients["ingredient_id"].unique())

    train_recipe_ingedient = recipe_ingredients[recipe_ingredients["recipe_id"].isin(core_train_rating["recipe_id"])]
    test_recipe_ingedient = recipe_ingredients[recipe_ingredients["recipe_id"].isin(core_test_rating["recipe_id"])]
    val_recipe_ingedient = recipe_ingredients[recipe_ingredients["recipe_id"].isin(core_val_rating["recipe_id"])]

    train = create_hetrodata(core_train_rating, ingredients.copy(), train_recipe_ingedient.copy(), user_label_encoder, recipe_label_encoder, ingredient_label_encoder, device)
    #test = create_hetrodata(core_test_rating, ingredients.copy(), test_recipe_ingedient.copy(), user_label_encoder, recipe_label_encoder, ingredient_label_encoder, device)
    val = create_hetrodata(core_val_rating, ingredients.copy(), val_recipe_ingedient.copy(), user_label_encoder, recipe_label_encoder, ingredient_label_encoder, device)

train