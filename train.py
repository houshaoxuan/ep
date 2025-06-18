import torch
import time
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from graphformer import GraphFormer

def load_dataset(dataset_name='MUTAG'):
    """加载TU数据集"""
    dataset = TUDataset(root='data/TU', name=dataset_name)

    # 打印数据集信息
    print(f"数据集: {dataset_name}")
    print(f"样本数量: {len(dataset)}")
    print(f"类别数量: {dataset.num_classes}")
    print(f"节点特征维度: {dataset.num_node_features}")

    # 检查是否存在边特征
    if dataset[0].edge_attr is not None:
        print(f"边特征维度: {dataset[0].edge_attr.size(-1)}")
    else:
        print("警告: 数据集不包含边特征")

    # 数据集划分
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, dataset.num_node_features, dataset.num_classes

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_loader, test_loader, input_dim, output_dim = load_dataset()

    # 初始化模型
    model = GraphFormer(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        output_dim=output_dim
    ).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        end_time = time.time()

        # 评估模型
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Epoch Time: {end_time - start_time:.2f}s')

if __name__ == "__main__":
    main()