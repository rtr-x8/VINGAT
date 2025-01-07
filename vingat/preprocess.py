from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd


class ScalarPreprocess:
    def __init__(self, x_dict):
        self.x_dict = x_dict
        self.standard_scaler_dict = {
            node: StandardScaler()
            for node in self.x_dict.keys()
        }

    def fit(self):
        for node, scaler in self.standard_scaler_dict.items():
            # 各ノードの属性 (x) に対して fit を適用
            self.standard_scaler_dict[node].fit(self.x_dict[node].cpu().numpy())
        return self

    def transform(self, x_dict):
        for node, val in x_dict.items():
            # 各ノードの属性 (x) に対して transform を適用
            x_dict[node].x = torch.tensor(
                self.standard_scaler_dict[node].transform(val.cpu().numpy()),
                dtype=val.dtype
            )
        return x_dict


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
    recip_ing.drop_duplicates(subset=['recipe_id', _key]).reset_index(drop=True)
    return recip_ing
