import pandas as pd
import os
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.io as io
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import numpy as np
import torch
from tqdm.notebook import tqdm

use_nutritions = ["niacin","fiber","sugars","sodium","carbohydrates","vitaminB6",
                  "calories","thiamin","fat","folate","caloriesFromFat","calcium",
                  "magnesium","iron","cholesterol","protein","vitaminA","potassium",
                  "saturatedFat","vitaminC"]

def parse_nutrient_json(json_dict):
  res = {}
  for un in use_nutritions:
    res.update({un: eval(json_dict).get(un).get("amount")})
  return res

def core_file_loader(directory_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(f"{directory_path}/core-data_recipe.csv", index_col=0),
        pd.read_csv(f"{directory_path}/core-data-train_rating.csv"),
        pd.read_csv(f"{directory_path}/core-data-test_rating.csv"),
        pd.read_csv(f"{directory_path}/core-data-valid_rating.csv")
    )


""" アイテム - 栄養素の一次保存データ """
def load_recipe_nutrients(directory_path: str, originarl_df: pd.DataFrame):
    if not os.path.isfile(f"{directory_path}/recipe_nutrients.csv"):

        __recipes = originarl_df.copy()
        __recipes[use_nutritions] = 0.0
        __recipes["nutritions"] = __recipes["nutritions"].to_dict()

        for idx, row in tqdm(__recipes.iterrows(), total = __recipes.shape[0]):
            nutri_text = row["nutritions"]
            nutri_val = parse_nutrient_json(nutri_text)
            for key, val in nutri_val.items():
                __recipes.loc[idx, key] = val

        __recipes.to_csv(f"{directory_path}/recipe_nutrients.csv")

    recipe_nutrients = pd.read_csv(f"{directory_path}/recipe_nutrients.csv", index_col=0)
    print("recipe_nutrients is loaded")
    return recipe_nutrients


"""食材の一時保存データ。"""
def load_ingredients(directory_path: str, originarl_df: pd.DataFrame):
    if not os.path.isfile(f"{directory_path}/ingredients.csv"):
        __recipes = originarl_df.copy()
        ingredients = set([""])
        for idx, row in tqdm(__recipes.iterrows(), total = __recipes.shape[0]):
            for ing in row["ingredients"].split("^"):
                ingredients.add(ing)
        ingredients.remove("")
        res = pd.DataFrame(ingredients, columns=["name"])
        res.index.name = "ingredient_id"
        res.to_csv(f"{directory_path}/ingredients.csv")

    ingredients = pd.read_csv(f"{directory_path}/ingredients.csv", index_col=0)
    print("ingredients is loaded")
    return ingredients
