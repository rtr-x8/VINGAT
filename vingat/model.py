import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, LGConv, HGTConv
import torch.nn as nn
import os


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
        return self.gnn(x_dict, edge_index_dict)


class RecommendationModel(nn.Module):
    def __init__(
        self,
        dropout_rate,
        device,
        hidden_dim,
        nutrient_dim=20
    ):
        super().__init__()

        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.device = device
        self.hidden_dim = hidden_dim

        self.user_norm = BatchNorm(hidden_dim)
        self.item_norm = BatchNorm(hidden_dim)
        self.intention_norm = BatchNorm(hidden_dim)
        self.image_norm = BatchNorm(hidden_dim)
        self.taste_norm = BatchNorm(hidden_dim)
        self.ing_norm = BatchNorm(hidden_dim)

        self.nutrient_projection = nn.Sequential(
            nn.Linear(nutrient_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # Contrastive caption and nutrient
        self.cl_nutrient_to_caption = ContrastiveLearning(hidden_dim, hidden_dim)

        # Fusion of ingredient and recipe
        self.ing_to_recipe = TasteGNN(hidden_dim)

        # HANConv layers
        self.fusion_gat = MultiModalFusionGAT(hidden_dim)

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):

        cl_nutirnent_x, cl_caption_x, cl_loss = self.cl_nutrient_to_caption(
            self.nutrient_projection(data["intention"].nutrient),
            data["intention"].x
        )
        data.x_dict.update({
            "intention": cl_caption_x,
        })
        data.x_dict.update({
            "user": self.user_norm(data["user"].x),
            "item": self.item_norm(data["item"].x),
            "intention": self.intention_norm(data["intention"].x),
            "image": self.image_norm(data["image"].x),
            "taste": self.taste_norm(data["taste"].x),
        })

        # Message passing
        data.x_dict.update({
            "taste": self.ing_to_recipe(data.x_dict, data.edge_index_dict)
        })
        """
        if not self.training:
            print("====")
            for k, v in data.edge_index_dict.items():
                print("-----")
                n1 = k[0]
                n2 = k[2]
                if not data.x_dict[n1].shape[0] >= v[0].max():
                    print(n1, data.x_dict[n1].shape[0] >= v[0].max(),
                          data.x_dict[n1].shape, v[0].max())
                if not data.x_dict[n2].shape[0] >= v[1].max():
                    print(n2, data.x_dict[n2].shape[0] >= v[1].max(),
                          data.x_dict[n2].shape, v[1].max())
        """
        fusion_out = self.fusion_gat(data.x_dict, data.edge_index_dict)
        data.x_dict.update(fusion_out)

        return data

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
