import pandas as pd

def core_file_loader(directory_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    return (
        pd.read_csv(f"{directory_path}/core-data_recipe.csv", index_col=0)
        pd.read_csv(f"{directory_path}/core-data-train_rating.csv")
        pd.read_csv(f"{directory_path}/core-data-test_rating.csv")
        pd.read_csv(f"{directory_path}/core-data-valid_rating.csv")
    )
