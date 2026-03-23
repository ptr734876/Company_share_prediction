import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, checkpoint_dir: str = None):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, learning_rate: float = 0.001,
              checkpoint_frequency: int = 1000, resume_from: str = None) -> dict:
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        start_epoch = 0
        if resume_from and self.checkpoint_dir:
            start_epoch = self._load_checkpoint(resume_from, optimizer)
            print(f"Resuming training from epoch {start_epoch}")
        
        print(f"\nStarting training for {epochs - start_epoch} epochs")
        print("-" * 50)
        
        train_loss = 0.0
        val_loss = 0.0
        
        for epoch in range(start_epoch, epochs):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch + 1, train_loss, val_loss, optimizer, is_best=True)
            
            if self.checkpoint_dir and (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(epoch + 1, train_loss, val_loss, optimizer, is_best=False)
                print(f"  ✓ Checkpoint saved on epoch {epoch + 1}")
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        print("-" * 50)
        print(f"Training completed!")
        print(f"  Final losses - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        
        if self.checkpoint_dir:
            self._save_checkpoint(epochs, train_loss, val_loss, optimizer, is_final=True)
            print(f"  Final model saved in {self.checkpoint_dir}")
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, criterion, optimizer) -> float:
        self.model.train()
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_epoch(self, loader: DataLoader, criterion) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(self.model(x), y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, 
                        optimizer, is_best: bool = False, is_final: bool = False):
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            
        elif is_final:
            path = os.path.join(self.checkpoint_dir, 'final_model.pt')
            torch.save(checkpoint, path)
            
        else:
            path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
            torch.save(checkpoint, path)
            
            meta_path = os.path.join(self.checkpoint_dir, 'checkpoint_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({'last_epoch': epoch}, f)
    
    def _load_checkpoint(self, checkpoint_type: str, optimizer) -> int:
        if checkpoint_type == 'best':
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        elif checkpoint_type == 'last':
            path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
        else:
            path = checkpoint_type
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Losses at checkpoint - Train: {checkpoint['train_loss']:.6f}, Val: {checkpoint['val_loss']:.6f}")
        
        return checkpoint['epoch']
    
    def get_best_model(self):
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        return self.model