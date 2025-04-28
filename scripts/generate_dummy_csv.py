# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:45:13 2025

@author: ryuse
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_data():
    num_users = 100
    num_merchants = 20
    num_trans = 1000
    num_transfers = 300

    base_time = datetime(2024, 1, 1)
    
    # 取引履歴（動的）
    transactions = []
    for i in range(num_trans):
        uid = random.randint(0, num_users - 1)
        mid = random.randint(0, num_merchants - 1)
        timestamp = base_time + timedelta(minutes=random.randint(0, 100000))
        label = random.choice([0, 1])  # 1 = デフォルトリスクあり
        transactions.append([uid, mid, timestamp.timestamp(), label])
    df_tx = pd.DataFrame(transactions, columns=['user_id', 'merchant_id', 'timestamp', 'label'])
    df_tx.to_csv('data/transactions.csv', index=False)

    # 送金履歴（静的）
    transfers = []
    for i in range(num_transfers):
        src = random.randint(0, num_users - 1)
        dst = random.randint(0, num_users - 1)
        if src != dst:
            transfers.append([src, dst])
    df_tf = pd.DataFrame(transfers, columns=['source', 'target'])
    df_tf.to_csv('data/transfers.csv', index=False)

if __name__ == '__main__':
    generate_data()
