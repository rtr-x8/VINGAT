import re

# setup.pyのパス
setup_file = "_version.py"

# setup.pyを読み込む
with open(setup_file, "r") as file:
    content = file.read()

# 現在のバージョンを取得
version_pattern = r"__version__=['\"]([0-9]+\.[0-9]+\.[0-9]+)['\"]"
current_version_match = re.search(version_pattern, content)
if not current_version_match:
    raise ValueError("Current version not found in setup.py")

current_version = current_version_match.group(1)

# バージョンを加算（パッチバージョンを加算）
major, minor, patch = map(int, current_version.split('.'))
new_version = f"{major}.{minor}.{patch + 1}"

# setup.pyのバージョンを更新
new_content = re.sub(version_pattern, f"__version__='{new_version}'", content)

# setup.pyを書き戻す
with open(setup_file, "w") as file:
    file.write(new_content)

print(f"Updated version from {current_version} to {new_version}")
