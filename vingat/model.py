import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn as nn
import os


class RepeatTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, output_dim):
        input_dim = tensor.size(1)
        if input_dim > output_dim:
            raise ValueError("input_dim が output_dim を超えることはできません。")

        # 必要な繰り返し回数を計算
        repeat_count = (output_dim + input_dim - 1) // input_dim  # 切り上げ
        # 自身を繰り返し結合して次元を増加
        repeated_tensor = tensor.repeat(1, repeat_count)
        # 必要な次元数にトリム
        return repeated_tensor[:, :output_dim]


# 栄養素情報によって強化された、選択理由テキスト
# 選択理由によって強化された栄養素情報
class ContrastiveLearning(nn.Module):
    def __init__(self, output_dim, temperature):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.nutrient_encoder = RepeatTensor()
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
        updated_nut = self.nutrient_encoder(nut_emb, self.output_dim)
        updated_cap = self.caption_encoder(text_emb)
        loss = self.info_nce_loss(updated_cap, updated_nut)
        return updated_cap, updated_nut, loss


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [
        ('ingredient', 'part_of', 'taste'),
        ('taste', 'contains', 'ingredient')
    ]

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.drop = DictDropout(dropout_rate)
        self.act = DictActivate()
        self.norm = DictBatchNorm(hidden_dim)
        self.gnn = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES)
        )
        """
        self.gnn = LGConv()
        """

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.drop(x_dict)
        out = self.norm(out)
        out = self.gnn(x_dict, edge_index_dict)
        out = self.act(out)
        return out
        """
        taste_x = x_dict['taste']
        taste_edge_index = edge_index_dict[('taste', 'contains', 'ingredient')]

        # LGConvの適用
        return self.gnn(taste_x, taste_edge_index)
        """


class DictActivate(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x_dict):
        return {
            k: self.act(v) for k, v in x_dict.items()
        }


class DictDropout(nn.Module):
    def __init__(self, dropout_rate, keys=[]):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.keys = keys

    def forward(self, x_dict):
        return {
            k: self.dropout(x_dict.get(k)) for k in self.keys
        }


class DictBatchNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = BatchNorm(hidden_dim)

    def forward(self, x_dict):
        return {
            k: self.norm(v) for k, v in x_dict.items()
        }


class MultiModalFusionGAT(nn.Module):
    """
    NODES = ['user', 'item', 'taste', 'intention', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('intention', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]
    """
    NODES = ['user', 'item', 'taste', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]

    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.drop = DictDropout(dropout_rate, self.NODES)
        self.act = DictActivate()
        self.norm = DictBatchNorm(hidden_dim)
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            heads=num_heads,
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.drop(x_dict)
        out = self.norm(out)
        out = self.gnn(x_dict, edge_index_dict)
        out = self.act(out)
        return out


def print_layer_outputs(model, input_data, max_elements=10, prefix=""):
    """
    モデルの各層の出力を表示する関数

    Args:
        model: モデル
        input_data: モデルへの入力データ
        max_elements: 表示する要素数の上限
    """
    for i, layer in enumerate(model.children()):
        input_data = layer(input_data)
        print(f"{prefix} Layer {i}: {layer.__class__.__name__}")
        # 出力の一部を表示 (最大 max_elements 個)
        print(input_data[:max_elements])
        print("-" * 20)


class RecommendationModel(nn.Module):
    def __init__(
        self,
        dropout_rate,
        device,
        hidden_dim,
        num_user: int,
        num_item: int,
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

        self.user_encoder = nn.Embedding(num_user, hidden_dim)
        self.item_encoder = nn.Embedding(num_item, hidden_dim)

        # Contrastive caption and nutrient
        """
        self.cl_with_caption_and_nutrient = nn.ModuleList()
        for _ in range(intention_layers):
            cl = ContrastiveLearning(hidden_dim, temperature)
            self.cl_with_caption_and_nutrient.append(cl)
        self.cl_dropout = DictDropout(dropout_rate, ["intention"])
        """

        # Fusion of ingredient and recipe
        self.ing_to_recipe = nn.ModuleList()
        for _ in range(sencing_layers):
            gnn = TasteGNN(hidden_dim, dropout_rate=dropout_rate)
            self.ing_to_recipe.append(gnn)

        self.taste_dropout = DictDropout(dropout_rate, ["taste"])

        # HANConv layers
        self.fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            gnn = MultiModalFusionGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            self.fusion_gnn.append(gnn)

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):

        data.set_x_dict("x", {
            "user": self.user_encoder(data["user"].id),
            "item": self.item_encoder(data["item"].id)
        })

        """
        cl_losses = []
        for cl in self.cl_with_caption_and_nutrient:
            caption_x, _, cl_loss = cl(data["intention"].x, data["intention"].nutrient)
            cl_losses.append(cl_loss)
            data.set_x_dict("x", {
                "intention": caption_x
            })
        cl_loss = torch.stack(cl_losses).mean()
        data.x_dict = self.cl_dropout(data.x_dict)
        """

        # Message passing
        for gnn in self.ing_to_recipe:
            data.set_x_dict("x", gnn(data.x_dict, data.edge_index_dict))

        data.set_x_dict("x", self.taste_dropout(data.x_dict))

        for gnn in self.fusion_gnn:
            data.set_x_dict("x", gnn(data.x_dict, data.edge_index_dict))

        data.set_x_dict("x", self.fusion_dropout(data.x_dict))

        return data   # , cl_loss

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        user_nodes = F.normalize(user_nodes, p=2, dim=1)
        recipe_nodes = F.normalize(recipe_nodes, p=2, dim=1)
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        # print_layer_outputs(self.link_predictor, edge_features)
        return self.link_predictor(edge_features)
