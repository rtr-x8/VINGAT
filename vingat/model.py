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
    def __init__(
        self,
        nutrient_input_dim,
        caption_input_dim,
        output_dim,
        temperature=0.5
    ):
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

    def forward(self, data):
        nutrient_emb = self.nutrient_encoder(data["intention"].nutrient)
        caption_emb = self.caption_encoder(data["intention"].caption)
        loss = self.loss(nutrient_emb, caption_emb)
        loss = self.loss(nutrient_emb, caption_emb)
        return (
            F.normalize(caption_emb, p=2, dim=1),
            F.normalize(nutrient_emb, p=2, dim=1),
            loss
        )


class TasteGNN(nn.Module):
    NODES = ['ingredient', 'taste']
    EDGES = [
        ('ingredient', 'part_of', 'taste'),
    ]

    def __init__(self, hidden_dim, dropout_rate, resisual_alpha):
        super().__init__()
        self.r_alpha = resisual_alpha
        self.gnn = HANConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.NODES, self.EDGES),
            dropout=dropout_rate,
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if k in self.NODES}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in self.EDGES}
        out = self.gnn(x_dict, edge_index_dict)

        for k, v in x_dict.items():
            if out.get(k) is None:
                out[k] = v
            else:
                out[k] = out[k] * (1-self.r_alpha) + v * self.r_alpha
        return out


class DictActivate(nn.Module):
    def __init__(self, device, keys=[]):
        super().__init__()
        self.acts = {
            k: nn.GELU().to(device) for k in keys
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


class DictLayerNorm(nn.Module):
    def __init__(self, hidden_dim, device, keys=[]):
        super().__init__()
        self.norms = {
            k: nn.LayerNorm(hidden_dim).to(device) for k in keys
        }

    def forward(self, x_dict):
        return {
            k: norm(x_dict.get(k)) for k, norm in self.norms.items()
        }


class MultiModalFusionGAT(nn.Module):
    def __init__(self, hidden_dim, num_heads, resisual_alpha, nodes, edges):
        super().__init__()
        self.r_alpha = resisual_alpha
        self.nodes = nodes
        self.edges = edges
        self.gnn = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=(self.nodes, self.edges),
            heads=num_heads,
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.gnn(
            {k: v for k, v in x_dict.items() if k in self.nodes},
            {k: v for k, v in edge_index_dict.items() if k in self.edges}
        )  # Only User, Item node

        for k, v in x_dict.items():
            if out.get(k) is None:
                out[k] = v
            else:
                out[k] = out[k] * (1-self.r_alpha) + v * self.r_alpha
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


class DictLayerNormForLayer(nn.Module):  # noqa F811 下の方で使ってるがコメントアウトしているので
    def __init__(self, node_types, hidden_dim):
        super().__init__()
        # 初期の軽いrescaling
        self.initial_rescaling = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })

        # メインのrescaling
        self.main_rescaling = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for node_type in node_types
        })

    def initial_forward(self, x_dict):
        return {
            node_type: self.initial_rescaling[node_type](x)
            for node_type, x in x_dict.items()
        }

    def main_forward(self, x_dict):
        return {
            node_type: self.main_rescaling[node_type](x)
            for node_type, x in x_dict.items()
        }


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
        user_encoder_dropout_rate: float,
        item_encoder_dropout_rate: float,
        intention_cl_after_dropout_rate: float,
        taste_gnn_dropout_rate: float,
        taste_gnn_after_dropout_rate: float,
        fusion_gnn_dropout_rate: float,
        fusion_gnn_after_dropout_rate: float,
        link_predictor_dropout_rate: float,
        link_predictor_leaky_relu_slope: float,
        sensing_gnn_resisual_alpha: float,
        fusion_gnn_resisual_alpha: float,
        is_abration_wo_cl: bool,
        is_abration_wo_taste: bool,
    ):
        super().__init__()
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.cl_loss_rate = cl_loss_rate
        if is_abration_wo_cl:
            self.cl_loss_rate = 0.0
        self.is_abration_wo_cl = is_abration_wo_cl
        self.is_abration_wo_taste = is_abration_wo_taste

        # Node Encoder
        self.user_encoder = nn.Sequential(
            nn.Embedding(num_user, user_encoder_low_rank_dim),
            nn.Linear(user_encoder_low_rank_dim, hidden_dim),
            # nn.ReLU(),
            nn.Dropout(p=user_encoder_dropout_rate)
        )
        self.item_encoder = nn.Sequential(
            nn.Embedding(num_item, item_encoder_low_rank_dim),
            nn.Linear(item_encoder_low_rank_dim, hidden_dim),
            # nn.ReLU(),
            nn.Dropout(p=item_encoder_dropout_rate)
        )
        self.image_encoder = LowRankLinear(input_image_dim, hidden_dim, rank=32)
        if not is_abration_wo_taste:
            self.taste_encoder = nn.Linear(input_cooking_direction_dim, hidden_dim)
        self.ingredient_encoder = nn.Linear(input_ingredient_dim, hidden_dim)

        # Contrastive caption and nutrient
        if not is_abration_wo_cl:
            self.intention_cl = nn.ModuleList([
                NutrientCaptionContrastiveLearning(
                    nutrient_input_dim=nutrient_dim,
                    caption_input_dim=input_vlm_caption_dim,
                    output_dim=hidden_dim,
                    temperature=temperature
                )
                for _ in range(intention_layers)
            ])
            self.intention_cl_after = nn.Sequential(
                # DictBatchNorm(hidden_dim, device, ["intention"]),
                DictActivate(device, ["intention"]),
                DictDropout(intention_cl_after_dropout_rate, device, ["intention"])
            )

        # Taste Level GAT
        self.sensing_gnn = nn.ModuleList([
            TasteGNN(hidden_dim,
                     dropout_rate=taste_gnn_dropout_rate,
                     resisual_alpha=sensing_gnn_resisual_alpha)
            for _ in range(sencing_layers)
        ])
        self.sensing_gnn_after = nn.Sequential(
            # DictBatchNorm(hidden_dim, device, ["taste", "ingredient"]),
            DictActivate(device, ["taste", "ingredient"]),
            DictDropout(taste_gnn_after_dropout_rate, device, ["taste"]),
        )

        # Fusion GAT
        fusion_nodes = ['user', 'item', 'taste', 'image']
        fusion_edges = [('taste', 'associated_with', 'item'),
                        ('image', 'associated_with', 'item'),
                        ('user', 'buys', 'item'),
                        ('item', 'bought_by', 'user')]
        if not is_abration_wo_cl:
            fusion_nodes.append("intention")
            fusion_edges.insert(1, ('intention', 'associated_with', 'item'))
        self.fusion_gnn = nn.ModuleList([
            MultiModalFusionGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                resisual_alpha=fusion_gnn_resisual_alpha,
                nodes=fusion_nodes,
                edges=fusion_edges
            )
            for _ in range(fusion_layers)
        ])
        self.fusion_gnn_after = nn.Sequential(
            # DictLayerNorm(hidden_dim, device, ["user", "item"]),
            DictActivate(device, ["user", "item"]),
            DictDropout(fusion_gnn_after_dropout_rate, device, ["user", "item"]),
        )

        # リンク予測のためのMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 中間層を広げる
            nn.LeakyReLU(link_predictor_leaky_relu_slope),  # 追加の活性化層
            nn.Dropout(link_predictor_dropout_rate),  # ドロップアウトを少し緩める
            nn.Linear(hidden_dim, 1),
        )

        # Layer Normalization
        ln_node_types = ["user", "item", "image", "ingredient", "intention"]
        if is_abration_wo_taste:
            ln_node_types.append("taste")
        self.layer_norm = DictLayerNormForLayer(
            node_types=ln_node_types,
            hidden_dim=hidden_dim
        )

        self.init_weights()

    def predict(self, user_nodes, recipe_nodes):
        user_nodes = F.normalize(user_nodes, p=2, dim=1)
        recipe_nodes = F.normalize(recipe_nodes, p=2, dim=1)
        edge_features = torch.cat([user_nodes, recipe_nodes], dim=1)
        logits = self.link_predictor(edge_features)

        # lの平均と標準偏差を計算
        # l_mean = torch.mean(l)
        # l_std = torch.std(l)
        # print(f"mean: {round(l_mean.item(), 4)}, std: {round(l_std.item(), 4)}")  # 平均と標準偏差を表示

        return torch.sigmoid(logits)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data: HeteroData):
        data.set_value_dict("x", {
            "user": self.user_encoder(data["user"].id),
            "item": self.item_encoder(data["item"].id),
            "image": self.image_encoder(data["image"].org),
            "ingredient": self.ingredient_encoder(data["ingredient"].org),
        })
        if not self.is_abration_wo_cl:
            data.set_value_dict("x", {
                "taste": self.taste_encoder(data["taste"].org)
            })

        # data.set_value_dict("x", self.layer_norm.initial_forward(data.x_dict))

        if not self.is_abration_wo_cl:
            cl_losses = []
            for cl in self.intention_cl:
                intention_x, _, cl_loss = cl(data)
                data.set_value_dict("x", {
                    "intention": intention_x
                })
                cl_losses.append(cl_loss)
                data.set_value_dict("x", self.intention_cl_after(data.x_dict))
            cl_loss = torch.stack(cl_losses).mean()
        else:
            cl_loss = torch.empty(0)

        for gnn in self.sensing_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
            data.set_value_dict("x", self.sensing_gnn_after(data.x_dict))

        for gnn in self.fusion_gnn:
            data.set_value_dict("x", gnn(data.x_dict, data.edge_index_dict))
            data.set_value_dict("x", self.fusion_gnn_after(data.x_dict))

        data.set_value_dict("x", self.layer_norm.main_forward(data.x_dict))

        return data, [
            {"name": "cl_loss", "loss": cl_loss, "weight": self.cl_loss_rate}
        ]
