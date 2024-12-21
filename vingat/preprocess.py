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
    merged = pd.merge(recip_ing, alternative_ing, on='ingredient_id')
    update_mask = (
        merged['score'].notna() &
        (merged['score'] > threshold) &
        (merged['ingredient_id'] != merged['alternative_ingredient'])
    )

    merged.loc[update_mask, 'ing_id'] = merged.loc[update_mask, 'alternative_ingredient']
    merged = merged.drop(columns=['alternative_ingredient', 'score'])
    merged = merged.drop_duplicates(subset=['recipe_id', 'ing_id']).reset_index(drop=True)
    return merged
