import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import HeteroData
import torch.nn as nn
import os
from vingat.loss import ContrastiveLoss


# 新しいCL
class NutrientCaptionContrastiveLearning(nn.Module):
    def __init__(self, nutrient_input_dim, caption_input_dim, output_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.nutrient_encoder = nn.Sequential(
            nn.Linear(nutrient_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.caption_encoder = nn.Sequential(
            nn.Linear(caption_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.cossine_similarity = nn.CosineSimilarity(dim=1)
        self.loss = ContrastiveLoss(temperature=temperature)

    def forward(self, caption, nutrient):
        nutrient_emb = self.nutrient_encoder(nutrient)
        caption_emb = self.caption_encoder(caption)
        loss = self.loss(nutrient_emb, caption_emb)
        return F.normalize(caption_emb, p=2, dim=1), F.normalize(nutrient_emb, p=2, dim=1), loss


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [
        ('ingredient', 'part_of', 'taste'),
    ]

    def __init__(self, hidden_dim, dropout_rate, device):
        super().__init__()
        self.norm = DictBatchNorm(hidden_dim, device, ["taste"])
        self.act = DictActivate(["taste"])
        self.drop = DictDropout(dropout_rate, device, ["taste"])
        self.gnn = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            dropout=dropout_rate,
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        ings = x_dict.get("ingredient")
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)
        out["ingredient"] = ings
        out["taste"] += x_dict["taste"]  # 残差結合
        out.update(self.norm(out))
        out.update(self.act(out))
        out.update(self.drop(out))
        return {
            k: v for k, v in out.items() if v is not None
        }


class DictActivate(nn.Module):
    def __init__(self, device, keys=[]):
        super().__init__()
        self.acts = {
            k: nn.ReLU().to(device) for k in keys
        }

    def forward(self, x_dict):
        return {
            k: act(x_dict.get(k)) for k, act in self.acts.items()
        }


class DictDropout(nn.Module):
    def __init__(self, dropout_rate, device, keys=[]):
        super().__init__()
        self.dropouts = {
            k: nn.Dropout(dropout_rate).to(device) for k in keys
        }

    def forward(self, x_dict):
        return {
            k: drop(x_dict.get(k)) for k, drop in self.dropouts.items()
        }


class DictBatchNorm(nn.Module):
    def __init__(self, hidden_dim, device, keys=[]):
        super().__init__()
        self.norms = {
            k: BatchNorm(hidden_dim).to(device) for k in keys
        }

    def forward(self, x_dict):
        return {
            k: norm(x_dict.get(k)) for k, norm in self.norms.items()
        }


class MultiModalFusionGAT(nn.Module):
    NODES = ['user', 'item', 'taste', 'intention', 'image']
    EDGES = [('taste', 'associated_with', 'item'),
             ('intention', 'associated_with', 'item'),
             ('image', 'associated_with', 'item'),
             ('user', 'buys', 'item'),
             ('item', 'bought_by', 'user')]

    def __init__(self, hidden_dim, num_heads, dropout_rate, device):
        super().__init__()
        self.norm = DictBatchNorm(hidden_dim, device, ["user", "item"])
        self.act = DictActivate(device, ["user", "item"])
        self.drop = DictDropout(dropout_rate, device, ["user", "item"])
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            heads=num_heads,
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.gnn(x_dict, edge_index_dict)
        for k, v in x_dict.items():  # 残差結合
            out[k] += v
        out.update(self.norm(out))
        out.update(self.act(out))
        out.update(self.drop(out))
        return {
            k: v for k, v in out.items() if v is not None
        }


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


class StaticEmbeddingEncoder():
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x):
        with torch.no_grad():
            return x[:, :self.output_dim]


class LowRankLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank, bias=True):
        super(LowRankLinear, self).__init__()
        self.u = nn.Linear(input_dim, rank, bias=False)
        self.v = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x):
        return self.v(self.u(x))


class __bk_RecommendationModel(nn.Module):
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

        self.user_encoder = nn.Embedding(num_user, hidden_dim)
        self.user_embedding_dropout = nn.Dropout(p=0.3)

        # 次元削減
        self.image_encoder = StaticEmbeddingEncoder(input_image_dim, hidden_dim)

        # 次元はこのまま使う
        # self.vlm_caption_encoder = StaticEmbeddingEncoder(input_vlm_caption_dim, hidden_dim)
        self.ingredient_encoder = StaticEmbeddingEncoder(input_ingredient_dim, hidden_dim)
        self.cooking_direction_encoder = StaticEmbeddingEncoder(input_cooking_direction_dim,
                                                                hidden_dim)

        # self.image_encoder = LowRankLinear(input_image_dim, hidden_dim, rank=64)

        # Contrastive caption and nutrient
        self.cl_with_caption_and_nutrient = nn.ModuleList()
        for _ in range(intention_layers):
            cl = NutrientCaptionContrastiveLearning(
                nutrient_dim, input_vlm_caption_dim, hidden_dim, temperature
            )
            self.cl_with_caption_and_nutrient.append(cl)
        self.cl_dropout = DictDropout(dropout_rate, device, ["intention"])
        self.cl_norm = BatchNorm(hidden_dim)
        self.cl_act = nn.ReLU()

        # Fusion of ingredient and recipe
        self.ing_to_recipe = nn.ModuleList()
        for _ in range(sencing_layers):
            gnn = TasteGNN(hidden_dim, dropout_rate=0.3, device=device)
            self.ing_to_recipe.append(gnn)
        self.after_sensing_norm = DictBatchNorm(hidden_dim, device, ["taste", "ingredient"])
        self.after_sensing_act = DictActivate(device, ["taste", "ingredient"])
        self.taste_dropout = DictDropout(dropout_rate, device, ["taste"])

        # HANConv layers
        self.fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            gnn = MultiModalFusionGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                device=device
            )
            self.fusion_gnn.append(gnn)
        self.after_fusion_norm = DictBatchNorm(
           hidden_dim, device, ["user", "item", "taste", "image", "intention"]
        )
        self.after_fusion_act = DictActivate(
           device, ["user", "item", "taste", "image", "intention"])
        self.fusion_dropout = DictDropout(dropout_rate, device, ["user", "item", "taste", "image"])

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
            "image": self.image_encoder(data["image"].x),
            # "intention": self.vlm_caption_encoder(data["intention"].caption),
            "ingredient": self.ingredient_encoder(data["ingredient"].x),
            "taste": self.cooking_direction_encoder(data["taste"].x)
        })
        data.set_value_dict("x", {
            "user": self.user_embedding_dropout(data.x_dict["user"])
        })

        cl_losses = []
        for cl in self.cl_with_caption_and_nutrient:
            intention_x, _, cl_loss = cl(data["intention"].caption, data["intention"].nutrient)
            cl_losses.append(cl_loss)
            data.set_value_dict("x", {
                "intention": intention_x
            })
        cl_loss = torch.stack(cl_losses).mean()
        data.set_value_dict("x", {
            "intention": self.cl_norm(data.x_dict["intention"])
        })
        data.set_value_dict("x", {
            "intention": self.cl_act(data.x_dict["intention"])
        })
        data.set_value_dict("x", self.cl_dropout(data.x_dict))

        # Message passing
        # Sensing
        for gnn in self.ing_to_recipe:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
        data.set_value_dict("x", self.after_sensing_norm(data.x_dict))
        data.set_value_dict("x", self.after_sensing_act(data.x_dict))
        data.set_value_dict("x", self.taste_dropout(data.x_dict))

        # Fusion
        for gnn in self.fusion_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
        data.set_value_dict("x", self.after_fusion_norm(data.x_dict))
        data.set_value_dict("x", self.after_fusion_act(data.x_dict))
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


class RecommendationModel(nn.Module):
    def __init__(
        self,
        dropout_rate: float,
        device: torch.device,
        hidden_dim: int,
        node_embeding_dimmention: int,
        num_user: int,
        num_item: int,
        nutrient_dim: int,
        num_heads: int,
        sencing_layers: int,
        fusion_layers: int,
        intention_layers: int,
        temperature: float,
        cl_loss_rate: float,
        input_image_dim: int,
        input_vlm_caption_dim: int,
        input_ingredient_dim: int,
        input_cooking_direction_dim: int,
        user_encoder_low_rank_dim: int,
        item_encoder_low_rank_dim: int,
    ):
        super().__init__()
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.dropout_rate = dropout_rate
        self.device = device
        self.hidden_dim = hidden_dim
        self.node_embeding_dimmention = node_embeding_dimmention
        self.num_user = num_user
        self.num_item = num_item
        self.nutrient_dim = nutrient_dim
        self.num_heads = num_heads
        self.sencing_layers = sencing_layers
        self.fusion_layers = fusion_layers
        self.intention_layers = intention_layers
        self.temperature = temperature
        self.cl_loss_rate = cl_loss_rate
        self.input_image_dim = input_image_dim
        self.input_vlm_caption_dim = input_vlm_caption_dim
        self.input_ingredient_dim = input_ingredient_dim
        self.input_cooking_direction_dim = input_cooking_direction_dim

        # Node Encoder
        self.user_encoder = nn.Sequential(
            nn.Embedding(num_user, user_encoder_low_rank_dim),
            nn.Linear(user_encoder_low_rank_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.item_encoder = nn.Sequential(
            nn.Embedding(num_item, item_encoder_low_rank_dim),
            nn.Linear(item_encoder_low_rank_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.image_encoder = LowRankLinear(input_image_dim, hidden_dim, rank=64)
        self.taste_encoder = nn.Linear(input_cooking_direction_dim, hidden_dim)
        self.intention_encoder = nn.Linear(nutrient_dim, hidden_dim)

        """
        self.ingredient_encoder = nn.Linear(input_ingredient_dim, hidden_dim)
        self.cooking_direction_encoder = nn.Linear(input_cooking_direction_dim, hidden_dim)
        """
        # Taste Level GAT
        """
        self.ingredient_to_taste_gnn = nn.ModuleList()
        for _ in range(sencing_layers):
            self.ingredient_to_taste_gnn.append(
                TasteGNN(hidden_dim, dropout_rate=0.3, device=device)
            )
        self.ingredient_to_taste_gnn_after = nn.Sequential(
            DictBatchNorm(hidden_dim, device, ["taste", "ingredient"]),
            DictActivate(device, ["taste", "ingredient"]),
            DictDropout(dropout_rate, device, ["taste"]),
        )
        """

        # Fusion GAT
        fusion_nodes = ["user", "item", "taste", "image", "intention"]
        self.multi_modal_fusion_gnn = nn.ModuleList()
        for _ in range(fusion_layers):
            self.multi_modal_fusion_gnn.append(
                MultiModalFusionGAT(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    device=device
                )
            )
        self.multi_modal_fusion_gnn_after = nn.Sequential(
            DictBatchNorm(hidden_dim, device, fusion_nodes),
            DictActivate(device, fusion_nodes),
            DictDropout(dropout_rate, device, fusion_nodes),
        )

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def predict(self, user_nodes, recipe_nodes):
        # ユーザーとレシピの埋め込みを連結
        user_nodes = F.normalize(user_nodes, p=2, dim=1)
        recipe_nodes = F.normalize(recipe_nodes, p=2, dim=1)
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        return self.link_predictor(edge_features)

    def forward(self, data: HeteroData):
        data.set_value_dict("x", {
            "user": self.user_encoder(data["user"].id),
            "item": self.item_encoder(data["item"].id),
            "image": self.image_encoder(data["image"].org),
            "ingredient": self.ingredient_encoder(data["ingredient"].org),
            "taste": self.cooking_direction_encoder(data["taste"].org),
            "intention": self.intention_encoder(data["intention"].nutrient),
        })

        """
        for gnn in self.ingredient_to_taste_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
        data.set_value_dict("x", self.ingredient_to_taste_gnn_after(data.x_dict))
        """

        for gnn in self.multi_modal_fusion_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))

        return data, []
