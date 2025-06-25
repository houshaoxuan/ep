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
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler


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

batch_size = 10

# Split and sample dataset (reduce to 100 samples each for train and test)
test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=0.8, random_state=42)
train_ids = train_ids[:100]  # Sample 100 training samples
test_ids = test_ids[:50]    # Sample 100 test samples
train_loader = DataLoader(Subset(dataset, train_ids), num_workers=4, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=5, pin_memory=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.L1Loss(reduction="sum")

# Check for GPU availability
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
model.to(DEVICE)

# Define profiler schedule: wait 1 step, warmup 1 step, active 3 steps, repeat twice
# my_schedule = schedule(wait=1, warmup=1, active=3, repeat=2)


for epoch in range(3):  # Reduced from 10 to 5 epochs
    model.train()
    batch_loss = 0.0

    timing_info_total = {
        "Linear Transformation Time": 0.0,
        "Centrality Encoding Time": 0.0,
        "Spatial Encoding Time": 0.0,
        "Attention Layers Time": 0.0,
        "Final Transformation Time": 0.0,
        "Total Forward Pass Time": 0.0
    }

    batch_count = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

        batch = batch.to(DEVICE)
        y = batch.y
        optimizer.zero_grad()
        # with record_function("model_forward"):
        # output = global_mean_pool(model(batch), batch.batch)

        output, timing_info = model(batch)

        for key in timing_info_total:
          timing_info_total[key] += timing_info[key]

        batch_count += 1

        """ # 打印各阶段耗时
        print(f"Linear Transformation Time: {timing_info['Linear Transformation Time'] / (batch_size):.4f}s")
        print(f"Centrality Encoding Time: {timing_info['Centrality Encoding Time'] / (batch_size):.4f}s")
        print(f"Spatial Encoding Time: {timing_info['Spatial Encoding Time'] / (batch_size):.4f}s")
        print(f"Attention Layers Time: {timing_info['Attention Layers Time']/ (batch_size):.4f}s")
        print(f"Final Transformation Time: {timing_info['Final Transformation Time']/ (batch_size):.4f}s")
        print(f"Total Forward Pass Time: {timing_info['Total Forward Pass Time']/ (batch_size):.4f}s") """

    # 计算并打印平均时间
    print(f"\nEpoch {epoch+1} Average Timing Info:")
    for key in timing_info_total:
        avg_time = timing_info_total[key] / (batch_count * batch_size)
        print(f"{key.replace('_', ' ').title()}: {avg_time:.4f}s")
