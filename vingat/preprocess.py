from sklearn.preprocessing import StandardScaler
import torch


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
            self.standard_scaler_dict[node].fit(self.x_dict[node].x.numpy())
        return self

    def transform(self, x_dict):
        for node, val in x_dict.items():
            # 各ノードの属性 (x) に対して transform を適用
            x_dict[node].x = torch.tensor(
                self.standard_scaler_dict[node].transform(val.x.numpy()),
                dtype=val.x.dtype
            )
        return x_dict
