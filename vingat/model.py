import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import BatchNorm
import torch.nn as nn
import pandas as pd


class StaticEmbeddingLoader():
    def __init__(self, data: pd.DataFrame, dimention: int, device):
        self.data = data
        self.embedding_dim = dimention
        self.cols = [f"e_{i}" for i in range(dimention)]
        self.device = device

    def __call__(self, indices: torch.tensor):
        print(f"indices type: {type(indices)}")
        print(f"indices device: {indices.device}")
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, self.cols].values
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device
        )


class MultiModalAttentionFusion(nn.Module):
    def __init__(self, feature_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        # 各モダリティの特徴量を変換するための線形層
        self.image_projection = nn.Linear(feature_dim, feature_dim)
        self.taste_projection = nn.Linear(feature_dim, feature_dim)
        self.nutrition_projection = nn.Linear(feature_dim, feature_dim)

        # Multi-head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 最終的な特徴量を生成するためのFFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, taste_features, nutrition_features):
        # batch_size = image_features.size(0)

        # 各特徴量を変換
        image_proj = self.image_projection(image_features)
        taste_proj = self.taste_projection(taste_features)
        nutrition_proj = self.nutrition_projection(nutrition_features)

        # 特徴量を結合して attention の入力にする
        # (batch_size, 3, feature_dim)
        combined_features = torch.stack(
            [image_proj, taste_proj, nutrition_proj],
            dim=1
        )

        # Self-Attention
        attended_features, _ = self.multihead_attn(
            combined_features,
            combined_features,
            combined_features
        )

        # Add & Norm
        attended_features = self.layer_norm1(
            combined_features + self.dropout(attended_features)
        )

        # 特徴量の平均を取る
        # (batch_size, feature_dim)
        averaged_features = torch.mean(attended_features, dim=1)

        # Feed Forward Network
        output = self.ffn(averaged_features)

        # Add & Norm
        fused_features = self.layer_norm2(
            averaged_features + self.dropout(output)
        )

        return fused_features


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
        device="cpu"
    ):
        super().__init__()

        self.device = device

        hidden_dim = 128
        self.hidden_dim = hidden_dim

        # 埋め込み層をモデル内で定義
        self.user_embedding = nn.Embedding(num_users, hidden_dim, max_norm=1.0)
        # self.ingredient_embedding = nn.Embedding(num_ingredients, hidden_dim, max_norm=1.0)

        # レシピの特徴量を変換する線形層
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

        # 食材からレシピへのGAT
        self.ing_to_recipe = GATConv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            heads=2,
            concat=False,
            dropout=dropout_rate)

        # ユーザーとレシピ間のGAT
        self.user_recipe_gat = GATConv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            heads=2,
            concat=False,
            dropout=dropout_rate)

        self.multimodal_fusion_gat = GATConv(
            (hidden_dim, hidden_dim),
            hidden_dim,
            heads=2,
            concat=False,
            dropout=dropout_rate)

        # recipeの特徴量を自己集約
        # self.feature_fusion = MultiModalAttentionFusion(
        #  feature_dim=hidden_dim,
        #  num_heads=4,
        #  dropout=dropout_rate
        # )

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
        user_ids = data['user'].id.long()
        data['user'].x = self.user_embedding(user_ids)

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
