import torch
import torch.nn.functional as F
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


class CookingDirectionEncoder(nn.Module):
    def __init__(self, data: pd.DataFrame, dimention: int, device) -> None:
        self.device = device
        self.data = data
        self.sbert = TextEncoder(dimention, device)

    def forward(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, "cooking_directions"].values
        return torch.as_tensor(
            self.sbert.encode(values),
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


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), dim=1)  # 出力を正規化して、意味空間に投影
        return x


def contrastive_loss(z1, z2, temperature=0.5):
    # Cosine Similarityを計算
    similarity_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(similarity_matrix, labels)
    return loss


class ContrastiveLearning(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=0.5):
        super().__init__()
        self.encoder = ContrastiveEncoder(input_dim, output_dim)
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = contrastive_loss(z1, z2, self.temperature)
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
        recipe_cooking_directions,
        input_recipe_feature_dim=20,
        dropout_rate=0.3,
        device="cpu",
        hidden_dim=128
    ):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        # Encoders
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
        self.cooking_direction_encoder = CookingDirectionEncoder(
            recipe_cooking_directions,
            hidden_dim,
            device
        )

        # Contrastive caption and nutrient
        self.close_nutrient_to_caption = ContrastiveLearning()

        # Fusion of ingredient and recipe
        self.ing_to_recipe = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['ingredient', 'taste'],
                      [('ingredient', 'part_of', 'taste'),
                       ('taste', 'contains', 'ingredient')])
        )

        # HANConv layers
        self.han_conv = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(['user', 'item', 'ingredient', 'taste', 'intention', 'image'],
                      [('taste', 'associated_with', 'item'),
                       ('intention', 'associated_with', 'item'),
                       ('image', 'associated_with', 'item'),
                       ('user', 'buys', 'item'),
                       ('item', 'bought_by', 'user')])
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
        user_x = self.user_encoder(data['user'].user_id.long())
        visual_x = self.visual_encoder(data['image'].recipe_id.long())
        caption_x = self.visual_caption_encoder(data['image'].recipe_id.long())
        nutrient_x = self.nutrient_encoder(data['intention'].nutrient.float())
        ingredient_x = self.ingredient_embedding(
            data['ingredient'].ingredient_id.long())
        cooking_direction_x = self.cooking_direction_encoder(
            data["taste"].recipe_id.long())

        # update
        data["user"].x = user_x
        data["visual"].x = visual_x
        data["intention"].x = self.close_nutrient_to_caption(nutrient_x,
                                                             caption_x)
        data["taste"].x = self.ing_to_recipe(ingredient_x, cooking_direction_x)

        # Message passing
        data.x_dict = self.han_conv(data.x_dict, data.edge_index_dict)
        data.x_dict = {key: self.recipe_norm(x) for key, x in data.x_dict.items()}

        return data

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
