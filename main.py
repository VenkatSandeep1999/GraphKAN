import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from trafficdataset import TrafficGraphDataset, evaluate_model, hybrid_loss
from GCN_KAND import GCN_KAND 
from config import Config
from __init__ import get_train_test_data

def main():
    conf = Config("config_fig.yaml")

    data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe, \
    train_vehicle_type, train_engine_config, train_gen_weight, \
    test_xs, test_ys, test_arms, test_xp, test_xt, test_xe, \
    test_vehicle_type, test_engine_config, test_gen_weight = \
        get_train_test_data(conf, need_road_network_structure_matrix=True)

    output_filename = "test_vehicle_types_grid.txt"
    
    with open(output_filename, "w") as f:
        for s in range(test_vehicle_type.shape[0]):
            f.write(f"Sample {s}:\n")
            
            for n in range(test_vehicle_type.shape[1]):
                time_steps = test_vehicle_type[s, n, :, 0].astype(int)
                line = " ".join(map(str, time_steps))
                f.write(line + "\n")
            
            f.write("\n")

    print(f"[Analysis] Vehicle IDs saved in grid format to '{output_filename}'")
    print(test_vehicle_type[4][776][10][0])
    print('************** Train - Predict **********')
    print('train_xs:', train_xs.shape,  'test_xs:', test_xs.shape, 'train_xp:', train_xp.shape, 'test_xp:', test_xp.shape, 'test_xe:', test_xe.shape, 'train_ys:', train_ys.shape, 'test_ys:', test_ys.shape,'arm shape',arm_shape)
    print("train v shape", train_vehicle_type.shape)
    print("test_vehicle_type", test_vehicle_type.shape)

    train_hist = [train_xs] if not isinstance(train_xs, list) else train_xs
    test_hist = [test_xs] if not isinstance(test_xs, list) else test_xs
    
    if conf.use_externel:
        if conf.observe_p != 0: 
            train_hist.append(train_xp); test_hist.append(test_xp)
        if conf.observe_t != 0:
            train_hist.append(train_xt); test_hist.append(test_xt)
        if conf.observe_p != 0 or conf.observe_t != 0:
            train_hist.append(train_xe); test_hist.append(test_xe)

    train_dataset = TrafficGraphDataset(
        train_hist, train_vehicle_type, train_engine_config, train_gen_weight, train_ys, train_arms
    )
    test_dataset = TrafficGraphDataset(
        test_hist, test_vehicle_type, test_engine_config, test_gen_weight, test_ys, train_arms
    )

    train_loader = PyGDataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
    
    max_type_id = int(np.max([np.max(train_vehicle_type), np.max(test_vehicle_type)]))
    dyn_num_types = max_type_id + 1
    
    max_engine_id = int(np.max([np.max(train_engine_config), np.max(test_engine_config)]))
    dyn_num_engines = max_engine_id + 1
    
    max_weight_id = int(np.max([np.max(train_gen_weight), np.max(test_gen_weight)]))
    dyn_num_weights = max_weight_id + 1

    print(f"[System] Dynamic Embeddings Found: Types={dyn_num_types}, Engines={dyn_num_engines}, Weights={dyn_num_weights}")

    sample = train_dataset[0]
    hist_dim = sample.x_history.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Using Device: {device}")
    print(f"[System] History Input Dim: {hist_dim}")

    model = GCN_KAND(
        history_steps=hist_dim,
        num_veh_types=dyn_num_types,
        num_engines=dyn_num_engines,
        num_weights=dyn_num_weights,
        hidden_channels=96,
        out_channels=1,
        grid_min=-3.0,
        grid_max=3.0,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-5
    )

    criterion = hybrid_loss

    print(f"\n[System] Starting Multi-Branch GCN-KAND Training with HuberLoss...")
    start_time = time.time()
    best_mae = float('inf')
    grid_batch = next(iter(train_loader)).to(device)

    for epoch in range(conf.epochs):
        model.train()
        train_loss = 0
        valid_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(
                batch.x_history, 
                batch.x_type, 
                batch.x_engine, 
                batch.x_weight, 
                batch.edge_index
            )
            
            loss = criterion(out, batch.y)
            if torch.isnan(loss): 
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            valid_batches += 1

        scheduler.step()

        if epoch < (conf.epochs // 2) and epoch % 10 == 0:
            model.update_grids(
                grid_batch.x_history,
                grid_batch.x_type,
                grid_batch.x_engine,
                grid_batch.x_weight,
                grid_batch.edge_index
            )
            print(f"   [KAN] Grids updated at epoch {epoch}")

        val_mae, val_mape, val_rmse = evaluate_model(model, test_loader, device, data.min_max_scala)
        avg_loss = train_loss / valid_batches if valid_batches > 0 else 0

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_fusion_model.pth")
            print(f"Epoch [{epoch+1}/{conf.epochs}] | Loss: {avg_loss:.4f} | MAE: {val_mae:.4f} *BEST*, MAPE: {val_mape:.4f}, RMSE: {val_rmse:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{conf.epochs}] | Loss: {avg_loss:.4f} | MAE: {val_mae:.4f}, MAPE: {val_mape:.4f}, RMSE: {val_rmse:.4f}")

    print(f"\nTotal Training Time: {time.time() - start_time:.2f}s")
    print(f"Best Eval MAE: {best_mae:.4f}")

if __name__ == '__main__':
    main()
