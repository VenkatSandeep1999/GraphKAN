import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv 

from KANLinear import KANLinear 

class GCN_KAND(nn.Module):
    def __init__(
        self,
        history_steps: int,          
        num_veh_types: int,      
        num_engines: int ,       
        num_weights: int ,       
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.0,
        aggr: str = 'mean', 
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        num_grids: int = 8,
        bsplines: bool = True, 
        kan: bool = True,
        device: str = 'cpu'
    ):
        super(GCN_KAND, self).__init__()
        
        self.device = device
        self.dropout = dropout
        
        self.embed_type = nn.Embedding(num_veh_types, 3)    
        self.embed_engine = nn.Embedding(num_engines, 10)   
        self.embed_weight = nn.Embedding(num_weights, 5)    
        
        self.veh_feat_dim = 18
        self.norm_veh = nn.BatchNorm1d(self.veh_feat_dim)

        self.periodicity_kan = KANLinear(
            in_features=history_steps,
            out_features=hidden_channels, 
            grid_size=num_grids,
            grid_range=[grid_min, grid_max]
        )
        self.norm_periodicity = nn.BatchNorm1d(hidden_channels)

        fusion_input_dim = hidden_channels + self.veh_feat_dim
        
        self.fusion_block = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, aggr=aggr, normalize=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, aggr=aggr, normalize=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.pre_out_norm = nn.BatchNorm1d(hidden_channels)
        
        if kan:
            self.out_layer = KANLinear(
                in_features=hidden_channels,
                out_features=out_channels,
                grid_size=num_grids,
                grid_range=[grid_min, grid_max]
            )
        else:
            self.out_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_history, x_type, x_engine, x_weight, edge_index, ablate_periodicity=False):
        e_type = self.embed_type(x_type.long()).squeeze()
        e_engine = self.embed_engine(x_engine.long()).squeeze()
        e_weight = self.embed_weight(x_weight.long()).squeeze()
        
        if e_type.ndim == 1: e_type = e_type.unsqueeze(1)
        if e_engine.ndim == 1: e_engine = e_engine.unsqueeze(1)
        if e_weight.ndim == 1: e_weight = e_weight.unsqueeze(1)

        v_emb = torch.cat([e_type, e_engine, e_weight], dim=1)
        v_emb = self.norm_veh(v_emb)
        
        if ablate_periodicity:
            p_emb = torch.zeros(
                x_history.size(0),
                self.periodicity_kan.out_features,
                device=x_history.device
            )
        else:
            p_emb = self.periodicity_kan(x_history)
            p_emb = self.norm_periodicity(p_emb)
            p_emb = F.gelu(p_emb)

        combined = torch.cat([p_emb, v_emb], dim=1)
        x = self.fusion_block(combined)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.gelu(x) 
            x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pre_out_norm(x)
        out = self.out_layer(x)
        
        return out

    def update_grids(self, x_history, x_type, x_engine, x_weight, edge_index):
        if hasattr(self.periodicity_kan, 'update_grid'):
            self.periodicity_kan.update_grid(x_history)
            
        if hasattr(self.out_layer, 'update_grid'):
             with torch.no_grad():
                e_type = self.embed_type(x_type.long()).squeeze()
                e_engine = self.embed_engine(x_engine.long()).squeeze()
                e_weight = self.embed_weight(x_weight.long()).squeeze()
                
                if e_type.ndim == 1: e_type = e_type.unsqueeze(1)
                if e_engine.ndim == 1: e_engine = e_engine.unsqueeze(1)
                if e_weight.ndim == 1: e_weight = e_weight.unsqueeze(1)
                
                v_emb = self.norm_veh(torch.cat([e_type, e_engine, e_weight], dim=1))
                p_emb = F.gelu(self.norm_periodicity(self.periodicity_kan(x_history)))
                
                combined = torch.cat([p_emb, v_emb], dim=1)
                x = self.fusion_block(combined)
                
                for i, conv in enumerate(self.convs):
                    x = self.batch_norms[i](F.gelu(conv(x, edge_index)))
                
                final_in = self.pre_out_norm(x)
                self.out_layer.update_grid(final_in)
