import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv
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


class DictActivate(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x_dict):
        return {
            k: self.act(v) for k, v in x_dict.items()
        }


class DictDropout(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_dict):
        return {
            k: self.dropout(v) for k, v in x_dict.items()
        }


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [('ingredient', 'part_of', 'taste')]

    def __init__(self, hidden_dim):
        super().__init__()
        self.act = nn.GELU()
        self.drop = DictDropout(0.4)
        self.gnn = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        x_dict = self.drop(x_dict)
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)

        # ingredient側はNoneで返却される
        return {
            "taste": self.act(out["taste"])
            # "taste": out["taste"]
        }


class MultiModalFusionGAT(nn.Module):
    NODES = ['user', 'item', 'taste', 'intention', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('intention', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]

    def __init__(self, hidden_dim, num_heads=2):
        super().__init__()
        self.act = DictActivate()
        self.drop = DictDropout(0.4)
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            heads=num_heads
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        x_dict = self.drop(x_dict)
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)
        return self.act(out)
        # return out


class RecommendationModel(nn.Module):

    NODES = ['user', 'item', 'taste', 'intention', 'image', "ingredient"]

    def __init__(
        self,
        device,
        hidden_dim,
        nutrient_dim=20,
        sencing_layers=3,
        fusion_layers=3,
        intention_layers=3,
        num_heads=2,
        dropout_rate=0.4,
    ):
        super().__init__()

        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.device = device
        self.hidden_dim = hidden_dim

        self.nutrient_projection = nn.Sequential(
            nn.Linear(nutrient_dim, hidden_dim),
            nn.GELU(),
        )
        self.projection = nn.ModuleDict({
            node: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim),
            ) for node in self.NODES
        })

        # Contrastive caption and nutrient
        # self.cl_nutrient_to_caption = ContrastiveLearning(hidden_dim, hidden_dim)

        # Sensing GNN layers
        self.sensing_gnn = nn.ModuleList()
        for _ in range(sencing_layers):
            gnn = TasteGNN(hidden_dim)
            self.sensing_gnn.append(gnn)

        # MultiModal Fusion GNN layers
        self.fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            gnn = MultiModalFusionGAT(hidden_dim, num_heads)
            self.fusion_gnn.append(gnn)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):

        # Linear projection
        data.x_dict["intention"].nutrient = self.nutrient_projection(data["intention"].nutrient)
        data.x_dict = {
            node: self.projection[node](x.to(self.device))
            for node, x in data.x_dict.items()
        }

        # cl_nutirnent_x, cl_caption_x, cl_loss = self.cl_nutrient_to_caption(
        #     self.nutrient_projection(data["intention"].nutrient),
        #     data["intention"].x
        # )
        # data.x_dict.update({
        #     "intention": cl_caption_x,
        # })

        # Sensing GNN
        for gnn in self.sensing_gnn:
            data.x_dict.update(gnn(data.x_dict, data.edge_index_dict))

        # MultiModal Fusion GNN
        for gnn in self.fusion_gnn:
            data.x_dict.update(gnn(data.x_dict, data.edge_index_dict))

        return data

    def predict(self, user_nodes, recipe_nodes):
        # 内積計算は性能上がらないので、GNNの出力を連結してMLPに入力する
        # もしくは内積計算は出力形式が今と異なるかもしれない
        # return (user_nodes * recipe_nodes).sum(dim=1)
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
