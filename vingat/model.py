import torch
import torch.nn.functional as F
from torch_geometric.nn import LGConv, HGTConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn as nn
import os


# 栄養素情報によって強化された、選択理由テキスト
# 選択理由によって強化された栄養素情報
class ContrastiveLearning(nn.Module):
    def __init__(self, input_dim, output_dim, temperature):
        super().__init__()
        self.temperature = temperature
        self.nutrient_encoder = nn.Linear(input_dim, output_dim)
        self.caption_encoder = nn.Linear(output_dim, output_dim)

    def info_nce_loss(self, text_emb, nut_emb):
        batch_size = text_emb.size(0)
        text_emb_norm = F.normalize(text_emb, p=2, dim=1)
        nut_emb_norm = F.normalize(nut_emb, p=2, dim=1)
        # (B, B)の類似度行列
        logits = torch.matmul(text_emb_norm, nut_emb_norm.t()) / self.temperature
        labels = torch.arange(batch_size).long().to(text_emb.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, text_emb, nut_emb):
        updated_nut = self.nutrient_encoder(nut_emb)
        updated_cap = self.caption_encoder(text_emb)
        loss = self.info_nce_loss(updated_cap, updated_nut)
        return updated_cap, updated_nut, loss


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


class MultiModalFusionGAT(nn.Module):
    NODES = ['user', 'item', 'taste', 'intention', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('intention', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]

    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.drop = DictDropout(dropout_rate)
        self.act = DictActivate()
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            heads=num_heads,
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)
        out = self.drop(out)
        out = self.act(out)
        return out


class RecommendationModel(nn.Module):
    def __init__(
        self,
        dropout_rate,
        device,
        hidden_dim,
        nutrient_dim=20,
        num_heads=2,
        sencing_layers=10,
        fusion_layers=10,
        intention_layers=10,
        temperature=0.05,
    ):
        super().__init__()

        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.device = device
        self.hidden_dim = hidden_dim

        self.user_norm = BatchNorm(hidden_dim)
        self.item_norm = BatchNorm(hidden_dim)

        # Contrastive caption and nutrient
        self.cl_with_caption_and_nutrient = nn.ModuleList()
        for _ in range(intention_layers):
            cl = ContrastiveLearning(hidden_dim, hidden_dim, temperature)
            self.cl_with_caption_and_nutrient.append(cl)

        # Fusion of ingredient and recipe
        # self.ing_to_recipe = TasteGNN(hidden_dim)

        # HANConv layers
        """一旦コメントアウト
        self.fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            gnn = MultiModalFusionGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            self.fusion_gnn.append(gnn)
        """

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):

        # cl_nutirnent_x, cl_caption_x, cl_loss = self.cl_nutrient_to_caption(
        #     self.nutrient_projection(data["intention"].nutrient),
        #     data["intention"].x
        # )
        # data.x_dict.update({
        #     "intention": cl_caption_x,
        # })
        data.x_dict.update({
            "user": self.user_norm(data.x_dict.get("user")),
            "item": self.item_norm(data.x_dict.get("item")),
        })

        cl_losses = []
        for cl in self.cl_with_caption_and_nutrient:
            caption_x, _, cl_loss = cl(data["intention"].x, data["intention"].nutrient)
            cl_losses.append(cl_loss)
            data.x_dict.update({
                "intention": caption_x,
            })
        cl_loss = torch.stack(cl_losses).mean()

        # Message passing
        # data.x_dict.update({
        #     "taste": self.ing_to_recipe(data.x_dict, data.edge_index_dict)
        # })

        """一旦コメントアウト
        for gnn in self.fusion_gnn:
            data.x_dict.update(gnn(data.x_dict, data.edge_index_dict))
        """

        return data, cl_loss

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)
