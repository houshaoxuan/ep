from sklearn.metrics import accuracy_score

model.eval()
with torch.no_grad():
    out = model(dataset.x.to(device), dataset.edge_index.to(device))
    val_loss = criterion(out[val_idx], dataset.y[val_idx].to(device))
    test_loss = criterion(out[test_idx], dataset.y[test_idx].to(device))
    val_acc = accuracy_score(dataset.y[val_idx].cpu(), out[val_idx].argmax(dim=1).cpu())
    test_acc = accuracy_score(dataset.y[test_idx].cpu(), out[test_idx].argmax(dim=1).cpu())
    print(f"验证损失: {val_loss.item()}, 验证准确率: {val_acc}")
    print(f"测试损失: {test_loss.item()}, 测试准确率: {test_acc}")