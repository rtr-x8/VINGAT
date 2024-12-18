from setuptools import setup, find_packages
import os
import pkg_resources
import re


def find_version():
    version_file = os.path.join('vingat', '_version.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        version_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("バージョン情報が見つかりません。")


setup(
  name='vingat',  # パッケージ名
  version=find_version(),  # バージョン
  packages=find_packages(),  # サブディレクトリに含まれるモジュールを自動で探す
  install_requires=[
    str(r)
    for r in pkg_resources.parse_requirements(
      open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    )
  ],
)
