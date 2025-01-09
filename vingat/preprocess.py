from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
from typing import Tuple, List
from torch_geometric.data import HeteroData


class ScalarPreprocess:
    def __init__(self, mapping: List[Tuple[str, str]]):
        """
        mapping_dict: Dict[str, str]
            ノード名と属性の対応を示す辞書
        """
        self.mapping = mapping
        self.standard_scaler = {
            map: StandardScaler()
            for map in mapping
        }

    def fit(self, data: HeteroData):
        for node, attr in self.mapping:
            self.standard_scaler[(node, attr)].fit(
                data[node][attr].cpu().numpy()
            )
        return self

    def transform(self, data: HeteroData):
        for node, attr in self.mapping:
            print("node: ", node, "attr: ", attr)
            print(data)
            print(data.get(node, "-1"))
            print(data[node][attr])
            data[node][attr] = torch.tensor(
                self.standard_scaler[(node, attr)].transform(data[node][attr].cpu().numpy()),
                dtype=data[node][attr].dtype
            ).to(data[node][attr].device)
        return data


def filter_recipe_ingredient(
    recip_ing: pd.DataFrame,
    alternative_ing: pd.DataFrame,
    threshold: int
) -> pd.DataFrame:
    """
    レシピと食材のデータフレームを受け取り、代替食材の数が閾値を超えるレシピを返す
    """
    _key = 'ingredient_id'
    alternative = alternative_ing[alternative_ing["score"] > threshold]
    mapping_dict = dict(zip(alternative['alternative_ingredient'], alternative[_key]))
    recip_ing[_key] = recip_ing[_key].map(mapping_dict).fillna(recip_ing[_key]).astype(int)
    recip_ing = recip_ing.drop_duplicates(subset=['recipe_id', _key])
    return recip_ing
