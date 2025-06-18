from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# 添加白名单
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])

def load_arxiv_dataset():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    return {
        'data': data,
        'train_idx': split_idx['train'],
        'val_idx': split_idx['valid'],
        'test_idx': split_idx['test'],
        'num_classes': dataset.num_classes
    }