
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from metrics import rmse, mape, mae
class TrafficGraphDataset(torch.utils.data.Dataset):
    def __init__(self, history_data, veh_type, veh_engine, veh_weight, ys, arms):
        super().__init__()
        self.history = history_data
        self.veh_type = veh_type
        self.veh_engine = veh_engine
        self.veh_weight = veh_weight
        
        self.ys = ys[:, :-1, :] 
        self.num_nodes = self.ys.shape[1] 
        self.edge_index = self.build_edges_from_arms(arms)

    def build_edges_from_arms(self, arms):
        src_nodes, dst_nodes = [], []
        if len(arms.shape) == 3: arms = arms[0] 
        limit_nodes = min(len(arms), self.num_nodes)
        for src_id in range(limit_nodes):
            for dst_id in arms[src_id]:
                if dst_id < self.num_nodes:
                    src_nodes.append(src_id)
                    dst_nodes.append(dst_id)
        return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    def __len__(self):
        return self.ys.shape[0]

    def process_feature(self, feat, idx, is_history=False):
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat, dtype=torch.float32)
        
        f_sample = feat[idx]
        
        if f_sample.ndim >= 2:
            return f_sample[:self.num_nodes, ...].reshape(self.num_nodes, -1)
        else:
            return f_sample.flatten().unsqueeze(0).repeat(self.num_nodes, 1)

    def __getitem__(self, idx):
        h_feats = []
        for h in self.history:
            h_feats.append(self.process_feature(h, idx, is_history=True))
        x_history = torch.cat(h_feats, dim=1)

        x_type = self.process_feature(self.veh_type, idx)
        x_engine = self.process_feature(self.veh_engine, idx)
        x_weight = self.process_feature(self.veh_weight, idx)
        
        if x_type.shape[1] > 1: x_type = x_type[:, 0].unsqueeze(1)
        if x_engine.shape[1] > 1: x_engine = x_engine[:, 0].unsqueeze(1)
        if x_weight.shape[1] > 1: x_weight = x_weight[:, 0].unsqueeze(1)

        y = torch.tensor(self.ys[idx], dtype=torch.float32)
        
        data = Data(
            x_history=x_history,
            x_type=x_type,
            x_engine=x_engine,
            x_weight=x_weight,
            edge_index=self.edge_index,
            y=y
        )
        data.num_nodes = self.num_nodes
        return data


def evaluate_model(model, loader, device, scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x_history, 
                batch.x_type, 
                batch.x_engine, 
                batch.x_weight, 
                batch.edge_index
            )
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())

    pred_array = np.concatenate(preds, axis=0)
    real_array = np.concatenate(targets, axis=0)
    
    orig_shape = pred_array.shape
    try:
        pred_inv = scaler.inverse_transform(pred_array)
        real_inv = scaler.inverse_transform(real_array)
    except:
        pred_inv = scaler.inverse_transform(pred_array.reshape(-1, 1)).reshape(orig_shape)
        real_inv = scaler.inverse_transform(real_array.reshape(-1, 1)).reshape(orig_shape)

    return mae(pred_inv, real_inv), mape(pred_inv, real_inv), rmse(pred_inv, real_inv)

def hybrid_loss(pred, target):
        return 0.7 * nn.HuberLoss(delta=1.0)(pred, target) + \
            0.3 * nn.L1Loss()(pred, target)