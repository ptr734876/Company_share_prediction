import torch
import numpy as np
import pandas as pd
import json
import os
import pickle
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from .models import LSTMModel
from .data import DataLoader, StockDataset
from .trainer import ModelTrainer

@dataclass
class PredictionResult:
    timestamp: datetime
    predicted_close: float
    confidence_lower: float
    confidence_upper: float
    has_real_data: bool = False
    real_close: Optional[float] = None
    error_percentage: Optional[float] = None

class StockPredictor:
    def __init__(self, db_path: str, model_dir: str = None):
        self.db_path = db_path
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models')
        self.data_loader = DataLoader(db_path)
        
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.sequence_length = 60
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = {}
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _generate_model_name(self, symbol: str, start_date: datetime, end_date: datetime, db_name: str = None) -> str:
        if db_name is None:
            db_name = Path(self.db_path).stem
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        db_name_clean = re.sub(r'[^\w\-_]', '', db_name)
        
        return f"{symbol}_{start_str}-{end_str}_{db_name_clean}"
    
    def build_model(self, input_size: int = 5, hidden_size: int = 128, 
                   num_layers: int = 2, dropout: float = 0.2):
        self.model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout).to(self.device)
        self.model_config = {
            'input_size': input_size, 
            'hidden_size': hidden_size,
            'num_layers': num_layers, 
            'dropout': dropout,
            'sequence_length': self.sequence_length, 
            'architecture': 'LSTM'
        }
        return self.model
    
    def train(self, symbol: str, start_date: datetime, end_date: datetime, 
            epochs: int = 100, model_name: str = None,
            checkpoint_frequency: int = 1000):
        
        if model_name is None:
            model_name = self._generate_model_name(symbol, start_date, end_date)
        
        print(f"Training model: {model_name}")
        print(f"Period: {start_date.date()} - {end_date.date()}")
        print(f"Database: {Path(self.db_path).name}")
        
        df = self.data_loader.load_stock_data(symbol, start_date, end_date)
        
        dataset = StockDataset(df, self.sequence_length)
        train_size = int(len(dataset) * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        self.scaler_features, self.scaler_target = dataset.get_scalers()
        
        if not self.model:
            self.build_model()
        
        checkpoint_dir = os.path.join(self.model_dir, model_name)
        trainer = ModelTrainer(self.model, self.device, checkpoint_dir)
        
        history = trainer.train(
            train_loader, 
            val_loader, 
            epochs=epochs,
            checkpoint_frequency=checkpoint_frequency
        )
        
        self.model = trainer.get_best_model()
        
        training_meta = {
            'symbol': symbol,
            'train_start': start_date.isoformat(),
            'train_end': end_date.isoformat(),
            'db_path': self.db_path,
            'db_name': Path(self.db_path).name,
            'total_epochs': epochs,
            'completed_epochs': epochs,
            'history': history
        }
        self._save_model_config(model_name, training_meta)
        
        return history
    
    def resume_training(self, symbol: str, model_name: str,
                    new_start_date: datetime, new_end_date: datetime,
                    additional_epochs: int = 1000,
                    checkpoint_frequency: int = 1000):
        
        self.load_model(model_name, checkpoint_type='last')
        
        old_meta_path = os.path.join(self.model_dir, model_name, 'model_config.json')
        with open(old_meta_path, 'r') as f:
            old_meta = json.load(f)
        
        old_start = datetime.fromisoformat(old_meta['train_start'])
        old_end = datetime.fromisoformat(old_meta['train_end'])
        
        new_combined_start = min(old_start, new_start_date)
        new_combined_end = max(old_end, new_end_date)
        
        new_model_name = self._generate_model_name(
            symbol, 
            new_combined_start, 
            new_combined_end,
            Path(self.db_path).stem
        )
        
        print(f"Old model: {model_name}")
        print(f"New model: {new_model_name}")
        print(f"Extended period: {new_combined_start.date()} - {new_combined_end.date()}")

        print(f"Loading new data: {new_start_date.date()} - {new_end_date.date()}")
        df_new = self.data_loader.load_stock_data(symbol, new_start_date, new_end_date)
        
        all_start = min(old_start, new_start_date)
        all_end = max(old_end, new_end_date)
        df_all = self.data_loader.load_stock_data(symbol, all_start, all_end)
        
        dataset = StockDataset(df_all, self.sequence_length)
        dataset.scaler_features = self.scaler_features
        dataset.scaler_target = self.scaler_target
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        dataset.features = self.scaler_features.transform(df_all[feature_cols].values)
        dataset.targets = self.scaler_target.transform(df_all['close'].values.reshape(-1, 1))
        
        train_size = int(len(dataset) * 0.8)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        new_checkpoint_dir = os.path.join(self.model_dir, new_model_name)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        
        old_scaler_path = os.path.join(self.model_dir, model_name, 'scalers.pkl')
        new_scaler_path = os.path.join(new_checkpoint_dir, 'scalers.pkl')
        shutil.copy2(old_scaler_path, new_scaler_path)
        
        old_checkpoint_path = os.path.join(self.model_dir, model_name, 'last_checkpoint.pt')
        new_checkpoint_path = os.path.join(new_checkpoint_dir, 'last_checkpoint.pt')
        if os.path.exists(old_checkpoint_path):
            shutil.copy2(old_checkpoint_path, new_checkpoint_path)
            print(f"Checkpoint copied from {old_checkpoint_path}")
        
        trainer = ModelTrainer(self.model, self.device, new_checkpoint_dir)
        
        last_epoch = old_meta.get('completed_epochs', 10000)
        total_epochs = last_epoch + additional_epochs
    
        print(f"Fine-tuning on all data ({all_start.date()} - {all_end.date()}): {additional_epochs} epochs")
        history = trainer.train(
            train_loader, 
            val_loader, 
            epochs=total_epochs,
            checkpoint_frequency=checkpoint_frequency,
            resume_from='last' if os.path.exists(new_checkpoint_path) else None
        )
        
        updated_meta = {
            'symbol': symbol,
            'train_start': all_start.isoformat(),
            'train_end': all_end.isoformat(),
            'db_path': self.db_path,
            'db_name': Path(self.db_path).name,
            'previous_models': old_meta.get('previous_models', []) + [model_name],
            'total_epochs': old_meta.get('total_epochs', 0) + additional_epochs,
            'completed_epochs': old_meta.get('completed_epochs', 0) + additional_epochs,
            'history': history
        }
        
        self._save_model_config(new_model_name, updated_meta)
        
        print(f"Model updated and saved as: {new_model_name}")
        
        return history, new_model_name
    
    def predict(self, symbol: str, start: datetime, end: datetime, compare_with_real: bool = False) -> List[PredictionResult]:
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        lookback = start - timedelta(days=self.sequence_length * 2)
        df_history = self.data_loader.load_stock_data(symbol, lookback, start)
        
        df_real = pd.DataFrame()
        if compare_with_real:
            try:
                df_real = self.data_loader.load_stock_data(symbol, start, end)
                print(f"Real data loaded for comparison: {len(df_real)} days")
            except:
                print("No real data available for comparison")
                df_real = pd.DataFrame()
        
        results, current_data = [], df_history.copy()
        current_date = start
        
        while current_date <= end:
            if len(current_data) < self.sequence_length:
                break
            
            last_seq = current_data.tail(self.sequence_length)
            features = self.scaler_features.transform(last_seq[['open', 'high', 'low', 'close', 'volume']].values)
            
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model(x).cpu().numpy()
                prediction = self.scaler_target.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                mc_preds = []
                self.model.train()
                for _ in range(100):
                    with torch.no_grad():
                        pred = self.scaler_target.inverse_transform(
                            self.model(x).cpu().numpy().reshape(-1, 1)
                        )[0][0]
                        mc_preds.append(pred)
                
                std = np.std(mc_preds)
                margin = 1.96 * std
            
            real_close = None
            error_pct = None
            has_real = False
            
            if compare_with_real and not df_real.empty:
                mask = df_real['date'] == current_date.date()
                matching_rows = df_real[mask]
                
                if not matching_rows.empty:
                    real_close = float(matching_rows.iloc[0]['close'])
                    error_pct = abs(prediction - real_close) / real_close * 100
                    has_real = True
            
            results.append(PredictionResult(
                timestamp=current_date,
                predicted_close=float(prediction),
                confidence_lower=float(prediction - margin),
                confidence_upper=float(prediction + margin),
                has_real_data=has_real,
                real_close=real_close,
                error_percentage=error_pct
            ))
            
            new_row = pd.DataFrame([{
                'timestamp': current_date, 'date': current_date.date(),
                'open': prediction * 0.99, 'high': prediction * 1.01,
                'low': prediction * 0.98, 'close': prediction,
                'volume': current_data['volume'].mean()
            }])
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            current_date += timedelta(days=1)
        
        return results
    
    def load_model(self, model_name: str, checkpoint_type: str = 'best'):
        checkpoint_dir = os.path.join(self.model_dir, model_name)
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Model directory not found: {checkpoint_dir}")
        
        if checkpoint_type == 'best':
            model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        elif checkpoint_type == 'last':
            model_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
        elif checkpoint_type == 'final':
            model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        else:
            model_path = checkpoint_type
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        config_path = os.path.join(checkpoint_dir, 'model_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        
        self.sequence_length = self.model_config.get('sequence_length', 60)
        
        self.build_model(
            input_size=self.model_config.get('input_size', 5),
            hidden_size=self.model_config.get('hidden_size', 128),
            num_layers=self.model_config.get('num_layers', 2),
            dropout=self.model_config.get('dropout', 0.2)
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        scaler_path = os.path.join(checkpoint_dir, 'scalers.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scalers file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_features = scalers['features']
            self.scaler_target = scalers['target']
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def _save_model_config(self, model_name: str, meta: dict):
        checkpoint_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        config_path = os.path.join(checkpoint_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump({**self.model_config, **meta}, f, indent=2)
        
        scaler_path = os.path.join(checkpoint_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'features': self.scaler_features, 'target': self.scaler_target}, f)
        
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(self.model.state_dict(), best_model_path)
        
        if 'history' in meta:
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(meta['history'], f)
    
    def save_architecture(self, model_name: str):
        checkpoint_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        arch_path = os.path.join(checkpoint_dir, 'architecture.json')
        
        architecture_info = {
            "model_class": "LSTMModel",
            "config": self.model_config,
            "pytorch_version": torch.__version__,
            "created_at": datetime.now().isoformat(),
            "layers": []
        }
        
        if self.model is not None:
            for name, module in self.model.named_modules():
                if name:
                    architecture_info["layers"].append({
                        "name": name,
                        "type": module.__class__.__name__,
                        "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
                    })
        
        with open(arch_path, 'w') as f:
            json.dump(architecture_info, f, indent=2)
        
        return arch_path