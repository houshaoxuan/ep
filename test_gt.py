import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from model.torchgtgf import Graphormer
from torch_geometric.datasets import MoleculeNet
import os

# Assuming the Graphormer model code is already provided as above
# Paste the Graphormer model code here (GraphNodeFeature, GraphAttnBias, etc., and Graphormer class)
# For brevity, I'll assume it's imported or defined elsewhere

def smiles_to_graph(smiles, y):
    """Convert SMILES string to PyTorch Geometric graph data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_ids = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atom_ids, dtype=torch.long).view(-1, 1)

    # Get edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Compute in-degree and out-degree
    if edge_index.size(1) == 0:
        in_degree = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
        out_degree = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    else:
        in_degree = degree(edge_index[1], num_nodes=mol.GetNumAtoms())
        out_degree = degree(edge_index[0], num_nodes=mol.GetNumAtoms())
        in_degree = in_degree.long()
        out_degree = out_degree.long()

    # Placeholder for spatial_pos, edge_input, attn_edge_type, attn_bias
    # Simplified for ESOL; in practice, compute shortest path distances or use Graphormer's preprocessing
    num_nodes = mol.GetNumAtoms()
    spatial_pos = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    edge_input = torch.zeros(num_nodes, num_nodes, 1, 1, dtype=torch.long)
    attn_edge_type = torch.zeros(num_nodes, num_nodes, 1, dtype=torch.long)
    attn_bias = torch.zeros(num_nodes + 1, num_nodes + 1)

    # Target value
    y = torch.tensor([y], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        in_degree=in_degree,
        out_degree=out_degree,
        spatial_pos=spatial_pos,
        edge_input=edge_input,
        attn_edge_type=attn_edge_type,
        attn_bias=attn_bias,
        y=y,
        num_nodes=num_nodes,
    )
    return data

def load_esol_dataset(data_path):
    """Load ESOL dataset and convert to graph data."""
    df = pd.read_csv(data_path)
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['log_solubility'])
        if graph is not None:
            graphs.append(graph)
    return graphs

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            y_true.extend(data.y.cpu().numpy().tolist())
            y_pred.extend(out.cpu().numpy().tolist())
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return total_loss / len(loader.dataset), mse, r2

def main():
    # Hyperparameters
    n_layers = 6
    num_heads = 8
    hidden_dim = 128
    dropout_rate = 0.1
    input_dropout_rate = 0.1
    ffn_dim = 512
    edge_type = "multi_hop"
    multi_hop_max_dist = 5
    attention_dropout_rate = 0.1
    output_dim = 1
    batch_size = 32
    learning_rate = 2e-4
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MoleculeNet(root="./", name="ESOL")


    # Split dataset
    train_data, temp_data = train_test_split(dataset, train_size=0.8, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=0.5, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Graphormer(
        n_layers=n_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        intput_dropout_rate=input_dropout_rate,  # Note: 'intput' typo in original code
        ffn_dim=ffn_dim,
        dataset_name="ESOL",  # Custom dataset, use default atom/edge counts
        edge_type=edge_type,
        multi_hop_max_dist=multi_hop_max_dist,
        attention_dropout_rate=attention_dropout_rate,
        output_dim=output_dim,
    ).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_mse, val_r2 = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val R²: {val_r2:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_graphormer_esol.pt")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("best_graphormer_esol.pt"))
    test_loss, test_mse, test_r2 = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results: Loss: {test_loss:.4f} | MSE: {test_mse:.4f} | R²: {test_r2:.4f}")

if __name__ == "__main__":
    main()