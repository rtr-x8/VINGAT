import torch
import torch_geometric
import torch_scatter
import torch_sparse
import pyg_lib
import torchvision


def assert_package_versions():
    assert torch.__version__ == "2.4.1+cu121"
    assert torch_geometric.__version__ == "2.6.1"
    assert torch_scatter.__version__ == "2.1.2+pt24cu121"
    assert torch_sparse.__version__ == "0.6.18+pt24cu121"
    assert pyg_lib.__version__ == "0.4.0+pt24cu121"
    assert torchvision.__version__ == "0.19.1+cu121"
