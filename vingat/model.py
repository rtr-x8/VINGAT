import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv
from torch_geometric.nn.norm import BatchNorm
import torch.nn as nn
import os
from vingat.loss import SeparationLoss


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


class NutCaptionContrastiveLearning(nn.Module):
    """
    InfoNCE lossを用いたContrastive Learningのモジュール
    """
    def __init__(self, nutrient_dim, output_dim, temperature):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.nutrient_encoder = nn.Linear(nutrient_dim, output_dim)

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
        loss = self.info_nce_loss(text_emb, updated_nut)
        return text_emb, updated_nut, loss


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [
        ('ingredient', 'part_of', 'taste'),
        ('taste', 'contains', 'ingredient')
    ]

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.drop = DictDropout(dropout_rate, self.NODES)
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
        out = self.norm(x_dict)
        out = self.drop(out)
        out = self.gnn(out, edge_index_dict)
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
        out = self.norm(x_dict)
        out = self.drop(out)
        out = self.gnn(out, edge_index_dict)
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
        node_embeding_dimmention: int,
        num_user: int,
        num_item: int,
        nutrient_dim=20,
        num_heads=2,
        sencing_layers=10,
        fusion_layers=10,
        intention_layers=10,
        temperature=0.05,
        cl_loss=0.5,
        input_image_dim=1024,
        input_vlm_caption_dim=384,
        input_ingredient_dim=384,
        input_cooking_direction_dim=384,
    ):
        super().__init__()

        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.device = device
        self.hidden_dim = hidden_dim
        self.tiny_hidden_dim = node_embeding_dimmention
        self.cl_loss = cl_loss

        self.user_encoder = nn.Embedding(num_user, hidden_dim, max_norm=1)
        self.item_encoder = nn.Embedding(num_item, hidden_dim, max_norm=1)
        self.image_encoder = nn.Linear(input_image_dim, hidden_dim)
        self.vlm_caption_encoder = nn.Linear(input_vlm_caption_dim, hidden_dim)
        self.ingredient_encoder = nn.Linear(input_ingredient_dim, hidden_dim)
        self.cooking_direction_encoder = nn.Linear(input_cooking_direction_dim, hidden_dim)

        # visual
        self.separation_loss = SeparationLoss(reg_lambda=0.01)

        # Contrastive caption and nutrient
        self.cl_with_caption_and_nutrient = nn.ModuleList()
        for _ in range(intention_layers):
            cl = NutCaptionContrastiveLearning(nutrient_dim, hidden_dim, temperature)
            self.cl_with_caption_and_nutrient.append(cl)
        self.cl_dropout = DictDropout(dropout_rate, ["intention"])
        self.cl_norm = DictBatchNorm(hidden_dim)

        # Fusion of ingredient and recipe
        self.ing_to_recipe = nn.ModuleList()
        for _ in range(sencing_layers):
            gnn = TasteGNN(hidden_dim, dropout_rate=dropout_rate)
            self.ing_to_recipe.append(gnn)

        self.taste_dropout = DictDropout(dropout_rate, ["taste"])

        self.batch_norm = DictBatchNorm(hidden_dim)

        # HANConv layers
        self.fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            gnn = MultiModalFusionGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            self.fusion_gnn.append(gnn)

        self.fusion_dropout = DictDropout(dropout_rate, ["user", "item", "taste", "image"])

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):

        data.set_value_dict("x", {
            "user": self.user_encoder(data["user"].id),
            "item": self.item_encoder(data["item"].id),
            "image": self.image_encoder(data["image"].x),
            "intention": self.vlm_caption_encoder(data["intention"].caption),
            "ingredient": self.ingredient_encoder(data["ingredient"].x),
            "taste": self.cooking_direction_encoder(data["taste"].x)
        })

        cl_losses = []
        for cl in self.cl_with_caption_and_nutrient:
            intention_x, _, cl_loss = cl(data["intention"].caption, data["intention"].nutrient)
            cl_losses.append(cl_loss)
            data.set_value_dict("x", {
                "intention": intention_x
            })
        cl_loss = torch.stack(cl_losses).mean()
        data.set_value_dict("x", self.cl_dropout(data.x_dict))
        data.set_value_dict("x", self.cl_norm(data.x_dict))

        # Message passing
        for gnn in self.ing_to_recipe:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
        data.set_value_dict("x", self.taste_dropout(data.x_dict))

        data.set_value_dict("x", self.batch_norm(data.x_dict))

        for gnn in self.fusion_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
        data.set_value_dict("x", self.fusion_dropout(data.x_dict))

        return data, [
            {"name": "cl_loss", "loss": cl_loss, "weight": self.cl_loss}
        ]

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        user_nodes = F.normalize(user_nodes, p=2, dim=1)
        recipe_nodes = F.normalize(recipe_nodes, p=2, dim=1)
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        # print_layer_outputs(self.link_predictor, edge_features)
        return self.link_predictor(edge_features)
