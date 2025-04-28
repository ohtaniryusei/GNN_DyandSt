# train_mooc.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import JODIEDataset
from full_model import FullModel
from utils.data_loader_mooc import load_mooc_dynamic_dataset, build_static_graph_from_mooc  # これ後で作ります！

def train():

    dataset = JODIEDataset(root='/workspaces/GNN_DyandSt/data', name='MOOC')

    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # MOOC データセットは通常1つのグラフオブジェクトを含みます

    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
        
    # モデル定義
    in_dim = 4 # MOOCは4特徴量くらいしかない（確認すること）
    model = FullModel(in_dim=in_dim, out_dim=128)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Static Graph ---
    static_graph = build_static_graph_from_mooc(mooc_csv_path='/workspaces/GNN_DyandSt/data/mooc/raw/mooc.csv', similarity_threshold=0.8)

    # --- Dynamic Graph ---
    dynamic_dataset = load_mooc_dynamic_dataset(mooc_csv_path='/workspaces/GNN_DyandSt/data/mooc/raw/mooc.csv', num_snapshots=10, in_dim=in_dim)

    model.train()
    auc_scores = []

    for epoch in range(20):
        total_loss = 0.0
        epoch_aucs = []

        for snapshot in dynamic_dataset:
            x, edge_index, edge_attr, y = snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y

            # 静的エンコーディング
            h_static = model.static_encoder(static_graph.x, static_graph.edge_index)

            # 時間情報のダミー（あとでちゃんと作る）
            B = x.shape[0]
            N = 10
            t = torch.randint(0, 1000, (B,))
            t_prime = torch.randint(0, 1000, (B, N))
            semantic_feat = torch.randn(B, N, 8)

            y_pred = model(snapshot, h_static, t, t_prime, semantic_feat).squeeze()
            
            loss = criterion(y_pred, y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            auc = roc_auc_score(y.squeeze().cpu().numpy(), y_pred.detach().cpu().numpy())
            epoch_aucs.append(auc)
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | AUC: {sum(epoch_aucs)/len(epoch_aucs):.4f}")
        auc_scores.append(sum(epoch_aucs)/len(epoch_aucs))

    print(f"\n✅ 最終平均AUC: {sum(auc_scores)/len(auc_scores):.4f}")

if __name__ == '__main__':
    train()
