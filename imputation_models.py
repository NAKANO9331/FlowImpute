import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GATConv
import math
import os
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn

def get_model(model_type, dataset_name, in_dim, out_dim=None):

    if dataset_name == 'hz_4x4':
        hidden_dim = 256
        num_layers = 4
    else:
        hidden_dim = 256
        num_layers = 6
    
    if model_type == 'stgcn':
        return SimpleSTGCN(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_blocks=num_layers)
    elif model_type == 'gnn':
        return GNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'chebyshev':
        return ChebyshevGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'graphsage':
        return GraphSAGEGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'attention':
        return AttentionGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'residual':
        return ResidualGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'adaptive':
        return AdaptiveGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'multiscale':
        return MultiscaleGNN(in_dim, out_dim if out_dim is not None else in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.act = nn.ELU()
    def forward(self, x, edge_index):
        residual = None
        for i in range(self.num_layers - 1):
            out = self.gcn_layers[i](x, edge_index)
            out = self.bn[i](out)
            out = self.act(out)
            out = self.dropout(out)
            if residual is not None:
                out = out + residual  # Residual connection
            residual = out
            x = out
        x = self.gcn_layers[-1](x, edge_index)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, K=5):
        super().__init__()
        self.temporal = nn.Conv1d(in_channels, out_channels, kernel_size=K, padding=K//2)
        self.act = nn.LeakyReLU()
    def forward(self, x, edge_index=None):
        # x shape: (1, 80, 11) expected to convert to (1, 11, 80)
        if x.ndim == 3 and x.shape[1] == 80 and x.shape[2] == 11:
            x = x.permute(0, 2, 1)  # (1, 80, 11) -> (1, 11, 80)
        x = self.temporal(x)
        x = self.act(x)
        return x

class SimpleSTGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=None, num_blocks=3, K=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            STGCNBlock(in_dim if i == 0 else hidden_dim, hidden_dim, K=K) for i in range(num_blocks)
        ])
        self.fc = nn.Linear(hidden_dim, out_dim if out_dim is not None else in_dim)
    def forward(self, x, edge_index=None):
        out = x
        for block in self.blocks:
            out = block(out, edge_index)
        # out: (N, hidden_dim, 1) or (N, hidden_dim, T)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)  # (N, hidden_dim, 1) -> (N, hidden_dim)
        elif out.ndim == 3:
            out = out.mean(-1)    # (N, hidden_dim, T) -> (N, hidden_dim)
        x = self.fc(out)
        if x.ndim == 1:
            x = x[None, :]  # Ensure at least (1, F)
        return x

class ChebyshevGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, K=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.conv_layers.append(ChebConv(in_dim, hidden_dim, K=K))
        for _ in range(num_layers - 2):
            self.conv_layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.conv_layers.append(ChebConv(hidden_dim, out_dim, K=K))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn[i](x)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.conv_layers[-1](x, edge_index)
        return x

class GraphSAGEGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, aggregator='mean', dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.conv_layers.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        self.conv_layers.append(SAGEConv(hidden_dim, out_dim, aggr=aggregator))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn[i](x)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.conv_layers[-1](x, edge_index)
        return x

class AttentionGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        self.conv_layers.append(GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.conv_layers[-1](x, edge_index)
        return x

class ResidualGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        # First layer
        self.conv_layers.append(GCNConv(in_dim, hidden_dim))
        # Middle layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        # Last layer
        self.conv_layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        if in_dim != out_dim:
            self.input_proj = nn.Linear(in_dim, out_dim)
        else:
            self.input_proj = None
        
    def forward(self, x, edge_index):
        residual = x
        if self.input_proj is not None:
            residual = self.input_proj(x)
        
        for i in range(self.num_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn[i](x)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.conv_layers[-1](x, edge_index)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        
        return x

class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.attention_weights = nn.ParameterList()
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.conv_layers.append(GCNConv(in_dim, hidden_dim))
        self.attention_weights.append(nn.Parameter(torch.ones(1)))
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.attention_weights.append(nn.Parameter(torch.ones(1)))
        self.conv_layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            conv_out = self.conv_layers[i](x, edge_index)
            weighted_out = conv_out * torch.sigmoid(self.attention_weights[i])
            x = self.bn[i](weighted_out)
            x = self.act(x)
            x = self.dropout(x)
        
        x = self.conv_layers[-1](x, edge_index)
        return x

class MultiscaleGNN(nn.Module):
    """Multi-scale GCN module"""
    def __init__(self, in_dim, out_dim, hidden_dim=256, num_layers=4, scales=[1, 2], dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.scales = scales
        self.hidden_dim = hidden_dim
        self.conv_layers = nn.ModuleList()
        self.bn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        # First layer
        first_layer_convs = nn.ModuleList()
        for scale in scales:
            first_layer_convs.append(ChebConv(in_dim, hidden_dim, K=scale))
        self.conv_layers.append(first_layer_convs)
        
        # Middle layers
        for _ in range(num_layers - 2):
            layer_convs = nn.ModuleList()
            for scale in scales:
                layer_convs.append(ChebConv(hidden_dim, hidden_dim, K=scale))
            self.conv_layers.append(layer_convs)
        
        # Last layer
        last_layer_convs = nn.ModuleList()
        for scale in scales:
            last_layer_convs.append(ChebConv(hidden_dim, out_dim, K=scale))
        self.conv_layers.append(last_layer_convs)
        
        # Scale weights
        self.scale_weights = nn.ParameterList()
        for _ in range(num_layers):
            layer_weights = nn.ParameterList()
            for _ in scales:
                layer_weights.append(nn.Parameter(torch.ones(1)))
            self.scale_weights.append(layer_weights)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        
    def forward(self, x, edge_index):
        for layer_idx in range(self.num_layers):
            scale_outputs = []
            
            for scale_idx, scale in enumerate(self.scales):
                conv_out = self.conv_layers[layer_idx][scale_idx](x, edge_index)
                weighted_out = conv_out * torch.sigmoid(self.scale_weights[layer_idx][scale_idx])
                scale_outputs.append(weighted_out)
            x = torch.stack(scale_outputs, dim=0).mean(dim=0)
            if layer_idx < self.num_layers - 1:
                x = self.bn[layer_idx](x)
            x = self.act(x)
            x = self.dropout(x)
        
        return x 

def apply_model_imputation(data, mask_matrix, adj_matrix, model_type='stgcn', dataset_name='hz_4x4', train_data=None, val_data=None):
    data = data.copy()
    try:
        T, N, F = data.shape
    except Exception as e:
        print('[DEBUG] Error unpacking shape:', e)
        raise
    # Create model save directory
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    # Generate model filename
    model_filename = f"{model_type}_{dataset_name}_N{N}_F{F}.pth"
    model_path = os.path.join(model_save_dir, model_filename)
    # Use factory function to get model
    model = get_model(model_type, dataset_name, in_dim=F, out_dim=F)
    edge_index = []
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index.size(1) == 0:
        edge_index = torch.tensor([[i, i] for i in range(N)], dtype=torch.long).t().contiguous()
    # ====== Check if there is a pre-trained model ======
    if os.path.exists(model_path):
        # Output information only when loading for the first time
        if train_data is not None:
            print(f"     Found pre-trained model: {model_filename}")
            print(f"    Loading model...")
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        if train_data is not None:
            print(f"    Model loading completed")
    else:
        # ====== Model training ======
        gnn_types = ['stgcn', 'gnn', 'chebyshev', 'graphsage', 'attention', 'residual', 'adaptive', 'multiscale']
        if train_data is not None and model_type in gnn_types:
            print(f"    Starting training {model_type.upper()} model...")
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            loss_fn = nn.MSELoss()
            epochs = 500
            best_loss = float('inf')
            patience = 5
            patience_counter = 0
            best_model_state = None
            for epoch in range(epochs):
                optimizer.zero_grad()
                total_loss = 0
                num_batches = 0
                x = torch.tensor(train_data, dtype=torch.float32)  # (T, N, F)
                for t in range(T):
                    pred = model(x[t].unsqueeze(0), edge_index)
                    loss = loss_fn(pred.squeeze(0), x[t])
                    loss.backward()
                    total_loss += loss.item()
                    num_batches += 1
                optimizer.step()
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"      Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.6f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"       Early stopped at Epoch {epoch+1}, Best Loss: {best_loss:.6f}")
                        break
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            print(f"    Training completed, final Loss: {avg_loss:.6f}")
            print(f"    Saving model to: {model_path}")
            torch.save(model.state_dict(), model_path)
            print(f"    Model saving completed")
        elif train_data is None and model_type in gnn_types:
            print(f"    Warning: No training data, using randomly initialized model")
    model.eval()
    with torch.no_grad():
        if model_type == 'stgcn':
            window_size = min(5, T)
            for t in range(T):
                start_t = max(0, t - window_size + 1)
                window_data = data[start_t:t+1]
                if window_data.shape[0] < window_size:
                    padding = np.repeat(window_data[-1:], window_size - window_data.shape[0], axis=0)
                    window_data = np.concatenate([window_data, padding], axis=0)
                x = torch.tensor(window_data[-1], dtype=torch.float32).unsqueeze(0)
                pred = model(x, edge_index)
                pred = pred.squeeze(0).numpy()
                for mask_id, value in enumerate(mask_matrix):
                    if value != 1 and mask_id < N:
                        data[t, mask_id, :] = pred[mask_id, :]
        else:
            for t in range(T):
                x = torch.tensor(data[t], dtype=torch.float32).unsqueeze(0)
                pred = model(x, edge_index)
                pred = pred.squeeze(0).numpy()
                for mask_id, value in enumerate(mask_matrix):
                    if value != 1 and mask_id < N:
                        data[t, mask_id, :] = pred[mask_id, :]
    return data

def batch_imputation_with_all_models(data, mask_matrix, adj_matrix, dataset_name='hz_4x4', train_data=None, val_data=None, save_dir=None):
    import pickle
    model_types = ['stgcn', 'gnn', 'chebyshev', 'graphsage', 'attention', 'residual', 'adaptive', 'multiscale']
    results = {}
    for model_type in model_types:
        print(f"[Batch Imputation] Imputing with {model_type}...")
        imputed_data = None
        try:
            imputed_data = apply_model_imputation(
                data, mask_matrix, adj_matrix,
                model_type=model_type,
                dataset_name=dataset_name,
                train_data=train_data,
                val_data=val_data
            )
            results[model_type] = imputed_data
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{dataset_name}_{model_type}_imputed.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(imputed_data, f)
                print(f"[Batch Imputation] {model_type} imputation results saved to: {save_path}")
        except Exception as e:
            print(f"[Batch Imputation] {model_type} imputation failed: {e}")
    return results 