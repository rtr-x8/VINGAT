import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LGConv, HGTConv
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import pandas as pd
import os


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
        super().__init__()
        self.device = device
        self.data = data
        self.sbert = TextEncoder(dimention, device)

    def forward(self, indices: torch.tensor):
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices, "direction"].values
        return torch.as_tensor(
            self.sbert(values),
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
        _indices = indices.clone().detach().to("cpu").numpy()
        values = self.data.loc[_indices].values.reshape(-1)
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
        return z1, z2, loss


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [('ingredient', 'part_of', 'taste')]

    def __init__(self, hidden_dim):
        super().__init__()
        """
        self.gnn = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES)
        )
        """
        self.gnn = LGConv()

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)
        return out["taste"]
        """

        taste_x = x_dict['taste']
        taste_edge_index = edge_index_dict[('taste', 'contains', 'ingredient')]

        # LGConvの適用
        return self.gnn(taste_x, taste_edge_index)


class MultiModalFusionGAT(nn.Module):
    NODES = ['user', 'item', 'taste', 'intention', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('intention', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]

    def __init__(self, hidden_dim):
        super().__init__()
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)
        return out["user"], out["item"], out["taste"], out["intention"], out["image"]


class RecommendationModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_recipes,
        num_ingredients,
        recipe_image_vlm_caption_embeddings,
        ingredients_with_embeddings,
        recipe_image_embeddings,
        recipe_cooking_directions_embeddings,
        input_recipe_feature_dim,
        dropout_rate,
        device,
        hidden_dim,
        user_max
    ):
        super().__init__()

        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.device = device
        self.hidden_dim = hidden_dim

        # Encoders
        self.user_encoder = nn.Embedding(user_max, hidden_dim, max_norm=1.0)
        self.visual_encoder = StaticEmbeddingLoader(
            recipe_image_embeddings,
            hidden_dim,
            device
        )
        self.visual_caption_encoder = StaticEmbeddingLoader(
            recipe_image_vlm_caption_embeddings,
            hidden_dim,
            device
        )
        self.nutrient_encoder = nn.Linear(input_recipe_feature_dim, hidden_dim)
        self.ingredient_embedding = StaticEmbeddingLoader(  # TODO: to SBERT
            ingredients_with_embeddings,
            hidden_dim,
            device
        )
        self.cooking_direction_encoder = StaticEmbeddingLoader(
            recipe_cooking_directions_embeddings,
            hidden_dim,
            device
        )

        # Contrastive caption and nutrient
        self.cl_nutrient_to_caption = ContrastiveLearning(hidden_dim, hidden_dim)

        # Fusion of ingredient and recipe
        self.ing_to_recipe = TasteGNN(hidden_dim)

        # HANConv layers
        self.fusion_gat = MultiModalFusionGAT(hidden_dim)

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
        ingredient_x = self.ingredient_embedding(data['ingredient'].ingredient_id.long())
        cooking_direction_x = self.cooking_direction_encoder(data["taste"].recipe_id.long())
        cl_nutirnent_x, cl_caption_x, cl_loss = self.cl_nutrient_to_caption(nutrient_x, caption_x)

        # update
        data["user"].x = user_x
        data["visual"].x = visual_x
        data["intention"].x = cl_caption_x
        data["ingredient"].x = ingredient_x
        data["taste"].x = cooking_direction_x

        # Message passing
        data.x_dict["taste"] = self.ing_to_recipe(data.x_dict, data.edge_index_dict)
        data.x_dict["user"],
        data.x_dict["item"],
        data.x_dict["taste"],
        data.x_dict["intention"],
        data.x_dict["image"] = self.fusion_gat(data.x_dict, data.edge_index_dict)
        # data.x_dict = {key: self.recipe_norm(x) for key, x in data.x_dict.items()}

        return data

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
