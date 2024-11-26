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

""" アイテム - 食材の一時保存データ """
def load_recipe_ingredients(directory_path: str, originarl_df: pd.DataFrame):
    if not os.path.isfile(f"{directory_path}/recipe_ingredients.csv"):

        __recipes = originarl_df.copy()
        recipe_ingredients = pd.DataFrame([], columns=["recipe_id", "ingredient_id"])
        ingredients = load_ingredients(directory_path, originarl_df)
        all_ings = ingredients.values.reshape(-1)
        for recipe_id, recipe_row in tqdm(__recipes.iterrows(), total = __recipes.shape[0]):
            for recip_ing in recipe_row["ingredients"].split("^"):
                if recip_ing in all_ings:
                    recipe_ingredients = pd.concat([recipe_ingredients,
                                                    pd.DataFrame([[recipe_id, np.where(all_ings == recip_ing)[0][0]]],
                                                                columns=["recipe_id", "ingredient_id"])])
        recipe_ingredients.to_csv(f"{directory_path}/recipe_ingredients.csv")

    recipe_ingredients = pd.read_csv(f"{directory_path}/recipe_ingredients.csv", index_col=0)
    print("recipe_ingredients is loaded")
    return recipe_ingredients


"""レシピ名、調理工程、レシピ画像の一時保存データ"""
def load_recipe_cooking_directions(directory_path: str, originarl_df: pd.DataFrame):
    if not os.path.isfile(f"{directory_path}/recipe_cooking_directions.csv"):

        __recipes = originarl_df.copy()
        __recipes["cooking_directions"] = __recipes["cooking_directions"].to_dict()

        recipe_cooking_directions = pd.DataFrame([], index=__recipes.index, columns=["direction"])
        for i in tqdm(__recipes.index.to_list(), total = __recipes.shape[0]):
            txt = __recipes.at[i, "cooking_directions"]
            txt = eval(txt).get('directions')
            txts = txt.split("\n")
            txts = [t for t in txts if len(t) > 10]
            txt = "\n".join(txts)
            recipe_cooking_directions.at[i, "direction"] = txt
        recipe_cooking_directions.to_csv(f"{directory_path}/recipe_cooking_directions.csv")

    recipe_cooking_directions = pd.read_csv(f"{directory_path}/recipe_cooking_directions.csv", index_col=0)
    print("recipe_cooking_directions is loaded")
    return recipe_cooking_directions


""" 食材の埋込データ """
def load_ingredients_with_embeddings(directory_path: str, originarl_df: pd.DataFrame):
    if not os.path.isfile(f"{directory_path}/ingredients_with_embeddings.csv"):
        raise Exception("embedding file not found") # next cell, next next cell, next next next cell
    ingredients_with_embeddings = pd.read_csv(f"{directory_path}/ingredients_with_embeddings.csv", index_col=0)
    print("ingredients_with_embeddings is loaded")
    return ingredients_with_embeddings


""" レシピ画像の特徴データ """
def load_recipe_image_embeddings(directory_path: str, originarl_df: pd.DataFrame, device):
    if not os.path.isfile(f"{directory_path}/recipe_image_embeddings.csv"):
        cnn_model = models.resnet50(pretrained=True)
        cnn = nn.Sequential(
            *(list(cnn_model.children())[:-2]),  # 最後の平均プーリング層の手前まで取得
            nn.Conv2d(2048, 1024, kernel_size=1),  # 1x1畳み込み層でoutput_dim次元に
            nn.AdaptiveAvgPool2d((1, 1))  # 出力を1x1にプールしてoutput_dim次元ベクトルに変換
        ).to(device)

        # 評価モードに設定
        cnn.eval()
        for param in cnn.parameters():
            param.requires_grad = False

        # 画像の前処理を定義
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])

        recipe_ingredients = load_recipe_ingredients(directory_path, originarl_df)
        recipe_image_embeddings = pd.DataFrame([], index=recipe_ingredients["recipe_id"].unique(), columns=[f"e_{i}" for i in range(1024)])
        errors = []

        def process_image(recipe_id):
            """個別の画像の読み込みと特徴量抽出を行う"""
            try:
                image_path = f"{directory_path}/core-data-images/core-data-images/{recipe_id}.jpg"
                image = io.read_image(image_path).to(device)
                image = image_transform(image).unsqueeze(0).to(device)  # 前処理とデバイス移動

                with torch.no_grad():
                    output = cnn(image)

                # 特徴量をCPUに移動してフラット化
            except Exception as e:
                errors.append(recipe_id)
                return recipe_id, np.zeros(1024)

            return recipe_id, output.view(-1).cpu().numpy()

        # 画像をバッチ処理
        with ThreadPoolExecutor(max_workers=4) as executor:  # 並列処理のスレッド数を調整
            results = list(tqdm(executor.map(process_image, recipe_image_embeddings.index.tolist()),
                                total=len(recipe_image_embeddings.index.tolist()),
                                desc="Image Embedding gen"))
        for recipe_id, embedding in tqdm(results, desc="save..."):
            recipe_image_embeddings.loc[recipe_id] = embedding

        # 結果をCSVに保存
        recipe_image_embeddings.to_csv(f"{directory_path}/recipe_image_embeddings.csv")

    recipe_image_embeddings = pd.read_csv(f"{directory_path}/recipe_image_embeddings.csv", index_col=0)
    print("recipe_image_embeddings is loaded")
    return recipe_image_embeddings
