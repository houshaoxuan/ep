import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.utils.convert import to_networkx
from networkx import all_pairs_shortest_path
from torch import nn
import torch_geometric.nn as tgnn
from model.graphormer.model import Graphormer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool

# Load dataset
dataset = MoleculeNet(root="./", name="ESOL")

# Initialize model
model = Graphormer(
    num_layers=3,
    input_node_dim=dataset.num_node_features,
    node_dim=128,
    input_edge_dim=dataset.num_edge_features,
    edge_dim=128,
    output_dim=dataset[0].y.shape[1],
    n_heads=4,
    ff_dim=256,
    max_in_degree=5,
    max_out_degree=5,
    max_path_distance=5,
)

# Split dataset
test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=0.8, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=100)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=100)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.L1Loss(reduction="sum")

# Training and evaluation
# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Using device: {DEVICE}")
model.to(DEVICE)

for epoch in range(10):
    model.train()
    batch_loss = 0.0
    for batch in tqdm(train_loader):
        batch.to(DEVICE)
        y = batch.y
        optimizer.zero_grad()
        output = global_mean_pool(model(batch), batch.batch)
        loss = loss_function(output, y)
        batch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print("TRAIN_LOSS", batch_loss / len(train_ids))

    model.eval()
    batch_loss = 0.0
    for batch in tqdm(test_loader):
        batch.to(DEVICE)
        y = batch.y
        with torch.no_grad():
            output = global_mean_pool(model(batch), batch.batch)
            loss = loss_function(output, y)
        batch_loss += loss.item()
    print("EVAL LOSS", batch_loss / len(test_ids))