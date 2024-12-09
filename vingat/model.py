import torch
from torch_geometric.nn import HANConv, BatchNorm
from sentence_transformers import SentenceTransformer
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


class TextEncoder(nn.Module):
    def __init__(self, dimention: int, device) -> None:
        super().__init__()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                                         truncate_dim=dimention,
                                         device=device)
        self.device = device

    def forward(self, sentences: list):
        return torch.as_tensor(
            self.sbert.encode(sentences),
            dtype=torch.float32,
            device=self.device
        )


class VLMEncoder(nn.Module):
    def __init__(self, data: pd.DataFrame, dimention: int, device) -> None:
        super().__init__()
        self.data = data
        self.embedding_dim = dimention
        self.device = device
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                                         truncate_dim=dimention,
                                         device=device)

    def forward(self, indices: torch.tensor):
        values = self.data.loc[indices, "text"].values
        return torch.as_tensor(
            self.sbert.encode(values),
            dtype=torch.float32,
            device=self.device
        )


class ContrastiveLearning(nn.Module):
    def __init__(self, device, mergin: int = 1.0):
        self.device = device
        self.mergin = mergin

    def forward(self, sentences: list):


class SimpleContrastiveLearning(nn.Module):
    def __init__(self, margin=1.0):
        super(SimpleContrastiveLearning, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, tensor1, tensor2, label):
        # Cosine similarity between the two tensors
        similarity = self.cosine_similarity(tensor1, tensor2)
        # Compute the contrastive loss
        loss = self.loss_fn(similarity, torch.zeros_like(similarity), label)
        return loss


class RecommendationModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_recipes,
        num_ingredients,
        recipe_image_vlm_caption,
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

        self.user_encoder = nn.Embedding(num_users, hidden_dim, max_norm=1.0)
        self.visual_encoder = StaticEmbeddingLoader(
            recipe_image_embeddings,
            hidden_dim,
            device
        )
        self.visual_caption_encoder = VLMEncoder(
            recipe_image_vlm_caption,
            hidden_dim,
            device
        )
        self.nutrient_encoder = nn.Linear(input_recipe_feature_dim, hidden_dim)
        self.ingredient_embedding = StaticEmbeddingLoader(  # TODO: to SBERT
            ingredients_with_embeddings,
            hidden_dim,
            device
        )



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
        data['user'].x = self.user_encoder(data['user'].user_id.long())

        # 食材特徴量
        ingredient_ids = data['ingredient'].id.long()
        data['ingredient'].x = self.ingredient_embedding(ingredient_ids)

        # レシピの特徴量：x
        recipe_feature = self.recipe_linear(data["recipe"].x)
        data['recipe'].x = self.recipe_norm(recipe_feature)

        # レシピの特徴量：visual
        data['recipe'].visual_feature = self.visual_encoder(data["recipe"].id.long())

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
