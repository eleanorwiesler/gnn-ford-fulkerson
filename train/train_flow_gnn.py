# === train/train_flow_gnn.py ===
import torch
from torch_geometric.data import DataLoader
from models.flow_predictor import FlowPredictorGNN
from torch.nn.functional import l1_loss

# Assume custom dataset that yields PyG-style Data objects
from my_dataset import FlowGraphDataset

dataset = FlowGraphDataset(root='data/processed')
train_loader = DataLoader(dataset[:800], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset[800:], batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowPredictorGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = l1_loss(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

