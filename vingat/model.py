import torch
from torch_geometric.nn import HANConv, BatchNorm
import torch.nn as nn
import pandas as pd


class StaticEmbeddingLoader():
    def __init__(self, data: pd.DataFrame, dimention: int, device):
        self.data = data
        self.embedding_dim = dimention
        self.cols = [f"e_{i}" for i in range(dimention)]
        self.device = device

    def __call__(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, self.cols].values
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device
        )


class RecommendationModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_recipes,
        num_ingredients,
        ingredients_with_embeddings,
        recipe_image_embeddings,
        input_recipe_feature_dim=20,
        dropout_rate=0.3,
        device="cpu",
        hidden_dim=128
    ):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        self.user_embedding = nn.Embedding(num_users, hidden_dim, max_norm=1.0)
        self.recipe_linear = nn.Linear(input_recipe_feature_dim, hidden_dim)

        # 食材の埋め込みを取得する
        self.ingredient_embedding = StaticEmbeddingLoader(
            ingredients_with_embeddings,
            hidden_dim,
            device)

        # レシピ画像の埋め込み取得
        self.image_feature_loader = StaticEmbeddingLoader(
            recipe_image_embeddings,
            hidden_dim,
            device)

        # 食材の特徴量を変換する線形そう
        self.ingredient_linear = nn.Linear(1024, hidden_dim)

        # HANConv layers
        self.han_conv = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['user', 'recipe', 'ingredient', 'taste', 'intention', 'image'],
                      [('ingredient', 'to', 'taste'),
                       ('taste', 'to', 'recipe'),
                       ('intention', 'to', 'recipe'),
                       ('image', 'to', 'recipe'),
                       ('user', 'to', 'recipe'),
                       ('recipe', 'to', 'user')])
        )

        # Nromali
        self.recipe_norm = BatchNorm(hidden_dim)

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # ユーザー特徴量
        data['user'].x = self.user_embedding(data['user'].user_id.long())

        # 食材特徴量
        ingredient_ids = data['ingredient'].id.long()
        data['ingredient'].x = self.ingredient_embedding(ingredient_ids)

        # レシピの特徴量：x
        recipe_feature = self.recipe_linear(data["recipe"].x)
        data['recipe'].x = self.recipe_norm(recipe_feature)

        # レシピの特徴量：visual
        data['recipe'].visual_feature = self.image_feature_loader(data["recipe"].id.long())

        # レシピの特徴量：intention
        data['recipe'].intention_feature = torch.ones(
            (data["recipe"].num_nodes, self.hidden_dim)
        ).to(self.device)

        # レシピの特徴量：taste
        data['recipe'].taste_feature = torch.ones(
            (data["recipe"].num_nodes, self.hidden_dim)
        ).to(self.device)

        # レシピ特徴を登録

        # item_features = self.feature_fusion(
        #  image_feture,
        #  intention_feature,
        #  taste_feature
        # )

        # レシピの特徴量を更新
        data['recipe'].x = data['recipe'].x + self.ing_to_recipe(
            (data['ingredient'].x, data['recipe'].x),
            data['ingredient', 'used_in', 'recipe'].edge_index
        )

        # ユーザーとレシピ間の情報伝播
        recipe_out = self.user_recipe_gat(
            (data['user'].x, data['recipe'].x),
            data['user', 'buys', 'recipe'].edge_index
        )
        data['recipe'].x = recipe_out

        return data

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
