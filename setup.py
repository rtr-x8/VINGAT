from setuptools import setup, find_packages

setup(
    name='vingat',  # パッケージ名
    version='0.1',  # バージョン
    packages=find_packages(),  # サブディレクトリに含まれるモジュールを自動で探す
    install_requires=[  # 依存パッケージのリスト
        'numpy',
        'pandas',
    ],
)
