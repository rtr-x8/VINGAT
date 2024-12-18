"""
import torch


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = torch.min(x, dim=0).values
        self.max = torch.max(x, dim=0).values
        self.denom = self.max - self.min
        self.denom[self.denom == 0] = 1  # Prevent division by zero

    def transform(self, x):
        return (x - self.min) / self.denom


class BasePreprocess():
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class UserPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x: torch.Tensor):
        self.scaler.fit(self.data)

    def transform(self, x: torch.Tensor):
        return self.scaler.transform(x)


def userPreprocess(train, target):
    userPreprocess = UserPreprocess()
    userPreprocess.fit(train)
    return userPreprocess, userPreprocess.transform(target)


class RecipePreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x: torch.Tensor):
        self.scaler.fit(self.data)

    def transform(self, x: torch.Tensor):
        return self.scaler.transform(x)


class IngredientPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x: torch.Tensor):
        self.scaler.fit(self.data)

    def transform(self, x: torch.Tensor):
        return self.scaler.transform(x)


class IntentionPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x: torch.Tensor):
        self.scaler.fit(self.data)

    def transform(self, x: torch.Tensor):
        return self.scaler.transform(x)


class ImagePreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x: torch.Tensor):
        self.scaler.fit(self.data)

    def transform(self, x: torch.Tensor):
        return self.scaler.transform(x)
"""
