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

# Split and sample dataset (reduce to 100 samples each for train and test)
test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=0.8, random_state=42)
train_ids = train_ids[:100]  # Sample 100 training samples
test_ids = test_ids[:50]    # Sample 100 test samples
train_loader = DataLoader(Subset(dataset, train_ids), num_workers=4, batch_size=10, pin_memory=True)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=5, pin_memory=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.L1Loss(reduction="sum")

# Check for GPU availability
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
model.to(DEVICE)

# Define profiler schedule: wait 1 step, warmup 1 step, active 3 steps, repeat twice
# my_schedule = schedule(wait=1, warmup=1, active=3, repeat=2)

# Training and evaluation with profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    for epoch in range(3):  # Reduced from 10 to 5 epochs
        model.train()
        batch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            with record_function("data_to_device"):
                batch = batch.to(DEVICE)
                y = batch.y
                optimizer.zero_grad()
            # with record_function("model_forward"):
              # output = global_mean_pool(model(batch), batch.batch)
            model(batch)
"""             with record_function("loss_computation"):
                loss = loss_function(output, y)
                batch_loss += loss.item()
            with record_function("backward"):
                loss.backward()
            # with record_function("gradient_clipping"):
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with record_function("optimizer_step"):
                optimizer.step()
            prof.step() """

         #print(f"Epoch {epoch+1} TRAIN_LOSS: {batch_loss / len(train_ids)}")

"""         model.eval()
        batch_loss = 0.0
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} Evaluation"):
            with record_function("eval_data_to_device"):
                batch = batch.to(DEVICE)
                y = batch.y
            with torch.no_grad():
                with record_function("eval_model_forward"):
                    output = global_mean_pool(model(batch), batch.batch)
                with record_function("eval_loss_computation"):
                    loss = loss_function(output, y)
            batch_loss += loss.item()
        print(f"Epoch {epoch+1} EVAL_LOSS: {batch_loss / len(test_ids)}")

        prof.step() """

# Print key averages sorted by CUDA time
prof.export_chrome_trace("trace.json")
print(prof.key_averages(group_by_stack_n=4).table(sort_by="cpu_time_total", row_limit=5))