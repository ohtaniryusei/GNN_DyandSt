import pandas as pd
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def load_mooc_dynamic_dataset(mooc_csv_path, num_snapshots=10, in_dim=4):
    df = pd.read_csv(mooc_csv_path)

    # 時間順に並べる
    df = df.sort_values('timestamp')
    
    snapshots = []
    snapshot_size = len(df) // num_snapshots

    for i in range(num_snapshots):
        df_snap = df.iloc[i * snapshot_size : (i+1) * snapshot_size]

        users = df_snap['user_id'].unique()
        items = df_snap['item_id'].unique()
        num_users = max(users) + 1
        num_items = max(items) + 1
        num_nodes = num_users + num_items

        # edge_index
        edge_index = torch.tensor(df_snap[['user_id', 'item_id']].values.T, dtype=torch.long)
        edge_index[1] += num_users  # item idを後ろに押す

        # ノード特徴量（ここではランダムでもよいが、厳密にはdfのfeatureを集約する）
        x = torch.randn(num_nodes, in_dim)

        # ラベル（ユーザー側だけに付ける）
        y = torch.zeros(num_nodes, 1)
        for _, row in df_snap.iterrows():
            uid = int(row['user_id'])
            label = float(row['state_label'])
            y[uid] = label

        snapshots.append((x, edge_index, None, y.numpy()))

    features = [snap[0] for snap in snapshots]
    edge_indices = [snap[1] for snap in snapshots]
    edge_weights = [snap[2] for snap in snapshots]
    targets = [snap[3] for snap in snapshots]

    dynamic_dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets
    )

    return dynamic_dataset

# utils/static_graph_builder.py
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

def build_static_graph_from_mooc(mooc_csv_path, feature_start_col=4, similarity_threshold=0.8):
    """
    MOOCデータセットから静的グラフを作成する
    """
    # 1. データ読み込み
    df = pd.read_csv(mooc_csv_path)
    print(df.head(5))

    # 2. ユーザーごとの特徴量を平均
    user_features = df.groupby('user_id').mean().iloc[:, feature_start_col:].values
    print(user_features)

    # 3. コサイン類似度を計算
    similarity_matrix = cosine_similarity(user_features)

    # 4. 閾値以上のペアをstatic edgeにする
    edges = []
    num_users = user_features.shape[0]
    for i in range(num_users):
        for j in range(i+1, num_users):
            if similarity_matrix[i, j] >= similarity_threshold:
                edges.append([i, j])

    if len(edges) == 0:
        raise ValueError("⚠️ No static edges created. Try lowering the threshold.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]

    # ノード特徴も適当に用意（ランダム初期化でいい）
    x = torch.randn(num_users, user_features.shape[1])

    static_graph = Data(x=x, edge_index=edge_index)

    return static_graph
