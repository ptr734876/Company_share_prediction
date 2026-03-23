import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int, target_col: str = 'close'):
        self.sequence_length = sequence_length
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        features = data[feature_cols].values
        targets = data[target_col].values
        
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        
        self.features = self.scaler_features.fit_transform(features)
        self.targets = self.scaler_target.fit_transform(targets.reshape(-1, 1))
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def inverse_transform_target(self, scaled_value):
        return self.scaler_target.inverse_transform(scaled_value.reshape(-1, 1)).flatten()
    
    def get_scalers(self):
        return self.scaler_features, self.scaler_target

class DataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def load_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        query = '''
            SELECT timestamp, open, high, low, close, volume 
            FROM stock_prices 
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, start_date.isoformat(), end_date.isoformat()))
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df['date'] = df['timestamp'].dt.date
        return df